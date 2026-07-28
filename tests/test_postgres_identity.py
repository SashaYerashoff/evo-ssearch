from __future__ import annotations

import hashlib
import os
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from eva_db import DatabaseSettings, TransactionContext
from security import ALL_CHANNELS, Permission, Role
from security.permissions import ROLE_PERMISSIONS
from security.postgres_identity import (
    NIL_UUID,
    IdentityBootstrapConflict,
    IdentityRecord,
    PostgresIdentityRepository,
    SessionRecord,
)


class StubHasher:
    def __init__(self) -> None:
        self.hash_calls: list[str] = []
        self.verify_result = True
        self.rehash = False

    def hash(self, password: str) -> str:
        self.hash_calls.append(password)
        return f"hashed:{password}"

    def verify(self, password_hash: str, password: str) -> bool:
        return self.verify_result and password_hash == f"hashed:{password}"

    def needs_rehash(self, password_hash: str) -> bool:
        return self.rehash


class Result:
    def __init__(self, *, row=None, rows=None, rowcount=0) -> None:
        self._row = row
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return list(self._rows or [])


class ScriptedConnection:
    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.executions: list[tuple[str, tuple | None]] = []

    def execute(self, sql, params=None):
        compact = " ".join(str(sql).split())
        self.executions.append((compact, params))
        if not self.steps:
            raise AssertionError(f"unexpected SQL: {compact}")
        expected, result = self.steps.pop(0)
        if expected not in compact:
            raise AssertionError(f"expected {expected!r}, got {compact!r}")
        return result

    def assert_finished(self) -> None:
        if self.steps:
            raise AssertionError(f"unconsumed SQL steps: {self.steps!r}")


class FakePool:
    def __init__(self, *connections: ScriptedConnection) -> None:
        self.connections = list(connections)
        self.contexts: list[TransactionContext | None] = []

    @contextmanager
    def transaction(self, context=None, *, readonly=False):
        del readonly
        self.contexts.append(context)
        if not self.connections:
            raise AssertionError("unexpected transaction")
        yield self.connections.pop(0)


def identity(
    *,
    tenant_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
) -> IdentityRecord:
    return IdentityRecord(
        user_id=str(user_id or uuid.uuid4()),
        tenant_id=str(tenant_id or uuid.uuid4()),
        username="admin",
        display_name="Administrator",
        roles=frozenset({Role.ADMIN.value}),
        permissions=frozenset(permission.value for permission in Permission),
        allowed_channel_ids=frozenset({ALL_CHANNELS}),
        is_active=True,
    )


class IdentityRecordTests(unittest.TestCase):
    def test_records_are_immutable(self) -> None:
        record = identity()
        session = SessionRecord(
            session_id=str(uuid.uuid4()),
            identity=record,
            csrf_digest="digest",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        with self.assertRaises(FrozenInstanceError):
            record.username = "changed"
        with self.assertRaises(FrozenInstanceError):
            session.csrf_digest = "changed"


class RepositoryUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tenant_id = uuid.uuid4()
        self.user_id = uuid.uuid4()
        self.session_id = str(uuid.uuid4())
        self.hasher = StubHasher()

    def test_authenticate_is_case_insensitive_and_switches_actor_after_verify(self):
        expected = identity(tenant_id=self.tenant_id, user_id=self.user_id)
        connection = ScriptedConnection(
            [
                (
                    "lower(username) = lower(%s)",
                    Result(row=(self.user_id, "hashed:correct")),
                ),
                ("set_config('eva.actor_id'", Result(row=("",))),
                ("UPDATE iam.users", Result(rowcount=1)),
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        with patch.object(repository, "_load_identity", return_value=expected):
            actual = repository.authenticate(
                self.tenant_id,
                "  AdMiN  ",
                "correct",
            )

        self.assertEqual(actual, expected)
        self.assertEqual(pool.contexts[0].actor_id, NIL_UUID)
        self.assertEqual(connection.executions[0][1][1], "AdMiN")
        self.assertEqual(
            connection.executions[1][1],
            (str(self.user_id),),
        )
        connection.assert_finished()

    def test_authenticate_rejects_bad_password_without_switching_actor(self):
        self.hasher.verify_result = False
        connection = ScriptedConnection(
            [
                (
                    "lower(username) = lower(%s)",
                    Result(row=(self.user_id, "hashed:correct")),
                ),
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        self.assertIsNone(
            repository.authenticate(self.tenant_id, "admin", "wrong")
        )
        self.assertEqual(pool.contexts[0].actor_id, NIL_UUID)
        connection.assert_finished()

    def test_create_session_stores_only_binary_sha256_digests(self):
        record = identity(tenant_id=self.tenant_id, user_id=self.user_id)
        connection = ScriptedConnection(
            [("INSERT INTO iam.sessions", Result(rowcount=1))]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        session_id = repository.create_session(
            record,
            "bearer-secret",
            "csrf-secret",
            expiry,
            client_ip="127.0.0.1",
            user_agent="Chrome",
        )

        uuid.UUID(session_id)
        params = connection.executions[0][1]
        self.assertEqual(
            params[3],
            hashlib.sha256(b"bearer-secret").digest(),
        )
        self.assertEqual(
            params[4],
            hashlib.sha256(b"csrf-secret").digest(),
        )
        self.assertNotIn("bearer-secret", params)
        self.assertNotIn("csrf-secret", params)
        self.assertEqual(pool.contexts[0].actor_id, str(self.user_id))
        connection.assert_finished()

    def test_resolve_session_checks_validity_and_loads_authorization(self):
        expected = identity(tenant_id=self.tenant_id, user_id=self.user_id)
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        csrf_digest = hashlib.sha256(b"csrf").digest()
        connection = ScriptedConnection(
            [
                (
                    "s.revoked_at IS NULL",
                    Result(row=(uuid.UUID(self.session_id), self.user_id, csrf_digest, expiry)),
                ),
                ("set_config('eva.actor_id'", Result(row=("",))),
                ("UPDATE iam.sessions", Result(rowcount=1)),
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        with patch.object(repository, "_load_identity", return_value=expected):
            session = repository.resolve_session(self.tenant_id, "token")

        self.assertEqual(
            session,
            SessionRecord(
                session_id=self.session_id,
                identity=expected,
                csrf_digest=csrf_digest.hex(),
                expires_at=expiry,
            ),
        )
        query = connection.executions[0]
        self.assertIn("s.expires_at > clock_timestamp()", query[0])
        self.assertEqual(query[1][1], hashlib.sha256(b"token").digest())
        self.assertEqual(pool.contexts[0].actor_id, NIL_UUID)
        connection.assert_finished()

    def test_resolve_session_rejects_missing_expired_or_revoked_session(self):
        connection = ScriptedConnection(
            [("s.revoked_at IS NULL", Result(row=None))]
        )
        repository = PostgresIdentityRepository(
            FakePool(connection),
            self.hasher,
        )

        self.assertIsNone(repository.resolve_session(self.tenant_id, "token"))
        connection.assert_finished()

    def test_revoke_session_is_one_way_and_actor_attributed(self):
        connection = ScriptedConnection(
            [
                ("SELECT user_id FROM iam.sessions", Result(row=(self.user_id,))),
                ("set_config('eva.actor_id'", Result(row=("",))),
                ("SET revoked_at = clock_timestamp()", Result(rowcount=1)),
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        self.assertTrue(
            repository.revoke_session(self.tenant_id, "token", "logout")
        )
        self.assertEqual(pool.contexts[0].actor_id, NIL_UUID)
        self.assertEqual(
            connection.executions[2][1][2],
            hashlib.sha256(b"token").digest(),
        )
        connection.assert_finished()

    def test_list_sessions_returns_inventory_records(self):
        session_id = uuid.uuid4()
        created = datetime.now(timezone.utc) - timedelta(minutes=5)
        last_seen = datetime.now(timezone.utc)
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        connection = ScriptedConnection(
            [
                (
                    "FROM iam.sessions AS s JOIN iam.users AS u",
                    Result(
                        rows=[
                            (
                                session_id,
                                self.tenant_id,
                                self.user_id,
                                "operator",
                                created,
                                last_seen,
                                expires,
                                None,
                                None,
                                "127.0.0.1",
                                "Chrome",
                            )
                        ]
                    ),
                )
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        sessions = repository.list_sessions(
            self.tenant_id,
            actor_user_id=self.user_id,
            user_id=self.user_id,
        )

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].session_id, str(session_id))
        self.assertEqual(sessions[0].username, "operator")
        self.assertEqual(sessions[0].client_ip, "127.0.0.1")
        self.assertEqual(pool.contexts[0].actor_id, self.user_id)
        self.assertEqual(connection.executions[0][1][1], self.user_id)
        self.assertTrue(connection.executions[0][1][3])
        connection.assert_finished()

    def test_revoke_session_by_id_uses_actor_context(self):
        session_id = uuid.uuid4()
        connection = ScriptedConnection(
            [("UPDATE iam.sessions", Result(rowcount=1))]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        self.assertTrue(
            repository.revoke_session_by_id(
                self.tenant_id,
                session_id,
                actor_user_id=self.user_id,
                reason="device_lost",
            )
        )

        self.assertEqual(pool.contexts[0].actor_id, self.user_id)
        self.assertEqual(connection.executions[0][1][0], "device_lost")
        self.assertEqual(connection.executions[0][1][2], session_id)
        connection.assert_finished()

    def test_password_update_revokes_active_sessions(self):
        expected = identity(tenant_id=self.tenant_id, user_id=self.user_id)
        connection = ScriptedConnection(
            [
                ("SELECT 1 FROM iam.users", Result(row=(1,))),
                ("UPDATE iam.users", Result(rowcount=1)),
                ("UPDATE iam.sessions", Result(rowcount=2)),
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)

        with patch.object(repository, "_load_identity", return_value=expected):
            actual = repository.update_user(
                self.tenant_id,
                self.user_id,
                actor_user_id=self.user_id,
                password="new password 123",
            )

        self.assertEqual(actual, expected)
        self.assertEqual(
            connection.executions[2][1][0],
            "account_security_changed",
        )
        connection.assert_finished()

    def test_bootstrap_is_idempotent_only_for_same_admin_username(self):
        existing = identity(tenant_id=self.tenant_id, user_id=self.user_id)
        connection = ScriptedConnection(
            [
                ("pg_advisory_xact_lock", Result(row=(None,))),
                (
                    "ORDER BY lower(u.username), u.id",
                    Result(rows=[(self.user_id, "Admin")]),
                ),
            ]
        )
        pool = FakePool(connection)
        repository = PostgresIdentityRepository(pool, self.hasher)
        role_ids = {role: uuid.uuid4() for role in Role}

        with (
            patch.object(
                repository,
                "_seed_authorization_catalogue",
                return_value=role_ids,
            ),
            patch.object(repository, "_load_identity", return_value=existing),
        ):
            actual = repository.bootstrap_admin(
                self.tenant_id,
                "admin",
                "unused-password",
            )

        self.assertEqual(actual, existing)
        self.assertEqual(self.hasher.hash_calls, [])
        connection.assert_finished()

    def test_bootstrap_rejects_different_existing_admin(self):
        connection = ScriptedConnection(
            [
                ("pg_advisory_xact_lock", Result(row=(None,))),
                (
                    "ORDER BY lower(u.username), u.id",
                    Result(rows=[(self.user_id, "first-admin")]),
                ),
            ]
        )
        repository = PostgresIdentityRepository(
            FakePool(connection),
            self.hasher,
        )

        with (
            patch.object(
                repository,
                "_seed_authorization_catalogue",
                return_value={role: uuid.uuid4() for role in Role},
            ),
            self.assertRaises(IdentityBootstrapConflict),
        ):
            repository.bootstrap_admin(
                self.tenant_id,
                "second-admin",
                "password",
            )
        connection.assert_finished()

    def test_catalogue_seeds_every_permission_role_and_mapping(self):
        class CatalogueConnection:
            def __init__(self):
                self.executions = []

            def execute(self, sql, params=None):
                compact = " ".join(str(sql).split())
                self.executions.append((compact, params))
                if "RETURNING id" in compact:
                    return Result(row=(params[0],))
                return Result(rowcount=1)

        connection = CatalogueConnection()
        repository = PostgresIdentityRepository(FakePool(), self.hasher)

        role_ids = repository._seed_authorization_catalogue(
            connection,
            self.tenant_id,
        )

        permission_inserts = [
            item for item in connection.executions
            if "INSERT INTO iam.permissions" in item[0]
        ]
        role_inserts = [
            item for item in connection.executions
            if "INSERT INTO iam.roles" in item[0]
        ]
        mapping_inserts = [
            item for item in connection.executions
            if "INSERT INTO iam.role_permissions" in item[0]
        ]
        self.assertEqual(len(permission_inserts), len(Permission))
        self.assertEqual(len(role_inserts), len(Role))
        self.assertEqual(
            len(mapping_inserts),
            sum(len(permissions) for permissions in ROLE_PERMISSIONS.values()),
        )
        self.assertEqual(set(role_ids), set(Role))

    def test_user_lifecycle_validates_before_database_access(self):
        repository = PostgresIdentityRepository(FakePool(), self.hasher)

        with self.assertRaisesRegex(ValueError, "cannot deactivate"):
            repository.update_user(
                self.tenant_id,
                self.user_id,
                actor_user_id=self.user_id,
                is_active=False,
            )
        with self.assertRaisesRegex(ValueError, "cannot remove"):
            repository.update_user(
                self.tenant_id,
                self.user_id,
                actor_user_id=self.user_id,
                roles=[Role.VIEWER.value],
            )
        with self.assertRaisesRegex(ValueError, "at least 12"):
            repository.create_user(
                self.tenant_id,
                actor_user_id=self.user_id,
                username="viewer",
                password="short",
                roles=[Role.VIEWER.value],
            )

    def test_admin_identity_uses_all_channels_without_reading_grants(self):
        connection = ScriptedConnection(
            [
                (
                    "SELECT id, tenant_id, username, display_name, is_active",
                    Result(
                        row=(
                            self.user_id,
                            self.tenant_id,
                            "admin",
                            None,
                            True,
                            False,
                        )
                    ),
                ),
                ("SELECT r.name", Result(rows=[(Role.ADMIN.value,)])),
                (
                    "SELECT DISTINCT rp.permission_key",
                    Result(rows=[(Permission.STREAMS_VIEW.value,)]),
                ),
            ]
        )
        repository = PostgresIdentityRepository(FakePool(), self.hasher)

        record = repository._load_identity(
            connection,
            self.tenant_id,
            self.user_id,
        )

        self.assertEqual(record.allowed_channel_ids, frozenset({ALL_CHANNELS}))
        connection.assert_finished()

    def test_non_admin_identity_uses_only_explicit_channel_grants(self):
        connection = ScriptedConnection(
            [
                (
                    "SELECT id, tenant_id, username, display_name, is_active",
                    Result(
                        row=(
                            self.user_id,
                            self.tenant_id,
                            "operator",
                            "Operator",
                            True,
                            False,
                        )
                    ),
                ),
                ("SELECT r.name", Result(rows=[(Role.OPERATOR.value,)])),
                (
                    "SELECT DISTINCT rp.permission_key",
                    Result(rows=[(Permission.STREAMS_VIEW.value,)]),
                ),
                ("SELECT channel_id", Result(rows=[(7,), (42,)])),
            ]
        )
        repository = PostgresIdentityRepository(FakePool(), self.hasher)

        record = repository._load_identity(
            connection,
            self.tenant_id,
            self.user_id,
        )

        self.assertEqual(record.allowed_channel_ids, frozenset({7, 42}))
        self.assertNotIn(ALL_CHANNELS, record.allowed_channel_ids)
        connection.assert_finished()

    def test_non_admin_identity_can_use_explicit_all_channel_grant(self):
        connection = ScriptedConnection(
            [
                (
                    "SELECT id, tenant_id, username, display_name, is_active",
                    Result(
                        row=(
                            self.user_id,
                            self.tenant_id,
                            "operator",
                            "Operator",
                            True,
                            True,
                        )
                    ),
                ),
                ("SELECT r.name", Result(rows=[(Role.OPERATOR.value,)])),
                (
                    "SELECT DISTINCT rp.permission_key",
                    Result(rows=[(Permission.STREAMS_VIEW.value,)]),
                ),
            ]
        )
        repository = PostgresIdentityRepository(FakePool(), self.hasher)

        record = repository._load_identity(
            connection,
            self.tenant_id,
            self.user_id,
        )

        self.assertEqual(record.allowed_channel_ids, frozenset({ALL_CHANNELS}))
        connection.assert_finished()


@unittest.skipUnless(
    os.getenv("EVA_TEST_DATABASE_DSN"),
    "set EVA_TEST_DATABASE_DSN for live PostgreSQL identity tests",
)
class PostgreSQLIdentityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from eva_db import PsycopgPool

        cls.pool = PsycopgPool(
            DatabaseSettings(
                dsn=os.environ["EVA_TEST_DATABASE_DSN"],
                pool_min_size=0,
                pool_max_size=2,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pool.close()

    def test_bootstrap_authenticate_session_resolve_and_revoke(self):
        tenant_id = uuid.uuid4()
        repository = PostgresIdentityRepository(self.pool)
        admin = repository.bootstrap_admin(
            tenant_id,
            "PilotAdmin",
            "correct horse battery staple",
            "Pilot Administrator",
        )
        repeated = repository.bootstrap_admin(
            tenant_id,
            "pilotadmin",
            "ignored replacement password",
        )

        self.assertEqual(repeated.user_id, admin.user_id)
        self.assertEqual(admin.roles, frozenset({Role.ADMIN.value}))
        self.assertEqual(admin.allowed_channel_ids, frozenset({ALL_CHANNELS}))
        self.assertEqual(
            repository.authenticate(tenant_id, "PILOTADMIN", "wrong"),
            None,
        )
        authenticated = repository.authenticate(
            tenant_id,
            "pilotadmin",
            "correct horse battery staple",
        )
        self.assertEqual(authenticated, admin)

        token = "session-token-" + uuid.uuid4().hex
        csrf_token = "csrf-token-" + uuid.uuid4().hex
        expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
        repository.create_session(
            authenticated,
            token,
            csrf_token,
            expiry,
            client_ip="127.0.0.1",
            user_agent="identity-live-test",
        )
        resolved = repository.resolve_session(tenant_id, token)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.identity, admin)
        self.assertEqual(
            resolved.csrf_digest,
            hashlib.sha256(csrf_token.encode()).hexdigest(),
        )
        self.assertTrue(repository.revoke_session(tenant_id, token, "test"))
        self.assertIsNone(repository.resolve_session(tenant_id, token))
        self.assertFalse(repository.revoke_session(tenant_id, token, "test"))

    def test_create_update_and_revoke_user_lifecycle(self):
        tenant_id = uuid.uuid4()
        repository = PostgresIdentityRepository(self.pool)
        admin = repository.bootstrap_admin(
            tenant_id,
            "LifecycleAdmin",
            "correct horse battery staple",
            "Lifecycle Administrator",
        )

        operator = repository.create_user(
            tenant_id,
            actor_user_id=admin.user_id,
            username="operator-" + uuid.uuid4().hex[:8],
            password="operator password 123",
            display_name="Pilot Operator",
            roles=[Role.OPERATOR.value],
            allowed_channel_ids=[7, 42],
        )

        self.assertEqual(operator.roles, frozenset({Role.OPERATOR.value}))
        self.assertEqual(operator.allowed_channel_ids, frozenset({7, 42}))
        self.assertIn(Permission.PROBES_RUN.value, operator.permissions)
        self.assertEqual(
            repository.get_user_by_username(
                tenant_id,
                operator.username.upper(),
                actor_user_id=admin.user_id,
            ),
            operator,
        )
        self.assertIn(
            operator.user_id,
            {
                user.user_id
                for user in repository.list_users(
                    tenant_id,
                    actor_user_id=admin.user_id,
                )
            },
        )

        authenticated = repository.authenticate(
            tenant_id,
            operator.username,
            "operator password 123",
        )
        self.assertEqual(authenticated, operator)
        token = "operator-session-" + uuid.uuid4().hex
        csrf_token = "operator-csrf-" + uuid.uuid4().hex
        repository.create_session(
            authenticated,
            token,
            csrf_token,
            datetime.now(timezone.utc) + timedelta(minutes=10),
            client_ip="127.0.0.1",
            user_agent="identity-lifecycle-live-test",
        )
        self.assertIsNotNone(repository.resolve_session(tenant_id, token))
        active_sessions = repository.list_sessions(
            tenant_id,
            actor_user_id=admin.user_id,
            user_id=operator.user_id,
        )
        self.assertEqual(len(active_sessions), 1)
        self.assertEqual(active_sessions[0].username, operator.username)
        self.assertTrue(
            repository.revoke_session_by_id(
                tenant_id,
                active_sessions[0].session_id,
                actor_user_id=admin.user_id,
                reason="single_session_rotation",
            )
        )
        self.assertIsNone(repository.resolve_session(tenant_id, token))
        self.assertFalse(
            repository.revoke_session_by_id(
                tenant_id,
                active_sessions[0].session_id,
                actor_user_id=admin.user_id,
                reason="single_session_rotation",
            )
        )

        second_token = "operator-session-2-" + uuid.uuid4().hex
        repository.create_session(
            authenticated,
            second_token,
            "operator-csrf-2-" + uuid.uuid4().hex,
            datetime.now(timezone.utc) + timedelta(minutes=10),
            client_ip="127.0.0.1",
            user_agent="identity-lifecycle-live-test",
        )
        self.assertIsNotNone(repository.resolve_session(tenant_id, second_token))

        viewer = repository.update_user(
            tenant_id,
            operator.user_id,
            actor_user_id=admin.user_id,
            roles=[Role.VIEWER.value],
            allowed_channel_ids=[7],
            display_name="Pilot Viewer",
        )
        self.assertEqual(viewer.roles, frozenset({Role.VIEWER.value}))
        self.assertEqual(viewer.allowed_channel_ids, frozenset({7}))
        self.assertNotIn(Permission.PROBES_RUN.value, viewer.permissions)
        self.assertIsNone(repository.resolve_session(tenant_id, second_token))

        third_token = "operator-session-3-" + uuid.uuid4().hex
        repository.create_session(
            authenticated,
            third_token,
            "operator-csrf-3-" + uuid.uuid4().hex,
            datetime.now(timezone.utc) + timedelta(minutes=10),
            client_ip="127.0.0.1",
            user_agent="identity-lifecycle-live-test",
        )
        revoked = repository.revoke_user_sessions(
            tenant_id,
            operator.user_id,
            actor_user_id=admin.user_id,
            reason="live_test_rotation",
        )
        self.assertEqual(revoked, 1)
        self.assertIsNone(repository.resolve_session(tenant_id, third_token))

        inactive = repository.update_user(
            tenant_id,
            operator.user_id,
            actor_user_id=admin.user_id,
            is_active=False,
        )
        self.assertFalse(inactive.is_active)
        self.assertIsNone(
            repository.authenticate(
                tenant_id,
                operator.username,
                "operator password 123",
            )
        )

    def test_non_admin_user_can_have_all_channel_access(self):
        tenant_id = uuid.uuid4()
        repository = PostgresIdentityRepository(self.pool)
        admin = repository.bootstrap_admin(
            tenant_id,
            "ScopeAdmin",
            "correct horse battery staple",
            "Scope Administrator",
        )

        operator = repository.create_user(
            tenant_id,
            actor_user_id=admin.user_id,
            username="operator-all-" + uuid.uuid4().hex[:8],
            password="operator password 123",
            display_name="All Channel Operator",
            roles=[Role.OPERATOR.value],
            allowed_channel_ids=[ALL_CHANNELS],
        )

        self.assertEqual(operator.roles, frozenset({Role.OPERATOR.value}))
        self.assertEqual(operator.allowed_channel_ids, frozenset({ALL_CHANNELS}))
        self.assertIn(Permission.PROBES_RUN.value, operator.permissions)


if __name__ == "__main__":
    unittest.main()
