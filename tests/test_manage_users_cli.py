from __future__ import annotations

import os
import unittest
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from scripts import manage_users
from security import ALL_CHANNELS, Permission, Role


@dataclass(frozen=True)
class _Actor:
    user_id: str
    username: str
    permissions: frozenset[str]
    is_active: bool = True


class _ActorRepository:
    def __init__(self, actor: _Actor | None = None) -> None:
        self.actor = actor
        self.get_user_calls = []

    def get_user_by_username(self, tenant_id, username, **_kwargs):
        if self.actor and self.actor.username == username:
            return self.actor
        return None

    def get_user(self, tenant_id, user_id, *, actor_user_id):
        self.get_user_calls.append((tenant_id, user_id, actor_user_id))
        if self.actor and self.actor.user_id == str(user_id):
            return self.actor
        return None


class ManageUsersCliUnitTests(unittest.TestCase):
    def test_parse_roles_defaults_to_viewer_and_validates_known_roles(self):
        self.assertEqual(
            manage_users._parse_roles([]),
            [Role.VIEWER.value],
        )
        self.assertEqual(
            manage_users._parse_roles(["operator, viewer"]),
            [Role.OPERATOR.value, Role.VIEWER.value],
        )
        with self.assertRaises(ValueError):
            manage_users._parse_roles(["owner"])

    def test_parse_channels_accepts_comma_list_and_all_channels(self):
        self.assertEqual(
            manage_users._parse_channels("7, 42", all_channels=False),
            [7, 42],
        )
        self.assertEqual(
            manage_users._parse_channels("", all_channels=True),
            [ALL_CHANNELS],
        )
        with self.assertRaises(SystemExit):
            manage_users._parse_channels("0", all_channels=False)

    def test_password_prefers_environment_variable(self):
        with patch.dict(os.environ, {"EVA_USER_PASSWORD": "from-env"}, clear=False):
            self.assertEqual(
                manage_users._password(
                    "EVA_USER_PASSWORD",
                    prompt="Password: ",
                ),
                "from-env",
            )

    def test_actor_user_id_requires_active_user_manager(self):
        tenant_id = uuid.uuid4()
        actor_id = str(uuid.uuid4())
        actor = _Actor(
            user_id=actor_id,
            username="admin",
            permissions=frozenset({Permission.USERS_MANAGE.value}),
        )
        args = SimpleNamespace(actor_user_id="", actor_username="admin")

        self.assertEqual(
            manage_users._actor_user_id(
                _ActorRepository(actor),
                tenant_id,
                args,
            ),
            actor_id,
        )

        viewer = _Actor(
            user_id=str(uuid.uuid4()),
            username="viewer",
            permissions=frozenset({Permission.STREAMS_VIEW.value}),
        )
        with self.assertRaisesRegex(SystemExit, "users:manage"):
            manage_users._actor_user_id(
                _ActorRepository(viewer),
                tenant_id,
                SimpleNamespace(actor_user_id="", actor_username="viewer"),
            )

    def test_actor_user_id_must_exist_when_passed_by_uuid(self):
        tenant_id = uuid.uuid4()
        actor_id = uuid.uuid4()
        args = SimpleNamespace(actor_user_id=str(actor_id), actor_username="admin")
        repository = _ActorRepository()

        with self.assertRaisesRegex(SystemExit, "Admin actor was not found"):
            manage_users._actor_user_id(repository, tenant_id, args)

        self.assertEqual(
            repository.get_user_calls,
            [(tenant_id, actor_id, actor_id)],
        )


if __name__ == "__main__":
    unittest.main()
