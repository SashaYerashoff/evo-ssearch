import os
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from eva_db import DatabaseSettings, PsycopgPool, TransactionContext
from agent_security import (
    ApprovalConsumedError,
    ApprovalError,
    ApprovalRequiredError,
    PostgresPlanApprovalStore,
    ToolExecutionContext,
    ToolGateway,
    ToolPolicy,
    ToolRegistry,
    ToolRisk,
)


UTC = timezone.utc


@unittest.skipUnless(
    os.getenv("EVA_TEST_DATABASE_DSN"),
    "set EVA_TEST_DATABASE_DSN for live PostgreSQL approval tests",
)
class PostgresPlanApprovalStoreIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = PsycopgPool(
            DatabaseSettings(
                dsn=os.environ["EVA_TEST_DATABASE_DSN"],
                pool_min_size=0,
                pool_max_size=4,
            )
        )
        cls.pool.open()

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def setUp(self):
        self.tenant_id = str(uuid4())
        self.actor_id = str(uuid4())
        self.other_actor_id = str(uuid4())
        self.session_id = uuid4().hex
        self.calls = []
        self._insert_user(self.actor_id, "operator")
        self._insert_user(self.other_actor_id, "other")
        registry = ToolRegistry()
        registry.register(
            "apply_setting",
            self._record_call,
            ToolPolicy(
                required_permission="probe.write",
                risk=ToolRisk.WRITE,
                approval_required=True,
                approval_required_when=lambda arguments: arguments.get(
                    "preview",
                    True,
                )
                is not True,
                allowed_arguments=frozenset({"channel_id", "value", "preview"}),
                required_arguments=frozenset({"channel_id", "value"}),
                channel_required=True,
            ),
        )
        store = PostgresPlanApprovalStore(self.pool)
        self.gateway = ToolGateway(
            registry,
            plan_store=store,
            approval_store=store,
            clock=lambda: datetime.now(UTC),
        )
        self.context = ToolExecutionContext(
            actor_id=self.actor_id,
            tenant_id=self.tenant_id,
            permissions={"probe.write"},
            allowed_channel_ids={"7"},
            agent_session_id=self.session_id,
            request_id="request-1",
        )

    def tearDown(self):
        self.gateway.close()
        context = TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=self.actor_id,
        )
        with self.pool.transaction(context) as connection:
            connection.execute(
                "DELETE FROM agent.action_approvals WHERE tenant_id = %s",
                (self.tenant_id,),
            )
            connection.execute(
                "DELETE FROM agent.action_plans WHERE tenant_id = %s",
                (self.tenant_id,),
            )
            connection.execute(
                "DELETE FROM agent.sessions WHERE tenant_id = %s",
                (self.tenant_id,),
            )
            connection.execute(
                "DELETE FROM iam.users WHERE tenant_id = %s",
                (self.tenant_id,),
            )

    def _insert_user(self, user_id, username):
        context = TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=user_id,
        )
        with self.pool.transaction(context) as connection:
            connection.execute(
                """
                INSERT INTO iam.users (
                    id,
                    tenant_id,
                    username,
                    password_hash
                )
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, self.tenant_id, username, "argon2-test-hash"),
            )

    def _record_call(self, context, arguments):
        call = {"context": context, "arguments": arguments}
        self.calls.append(call)
        return call

    def test_apply_plan_uses_hashed_token_and_is_one_time(self):
        with self.assertRaises(ApprovalRequiredError):
            self.gateway.execute(
                "apply_setting",
                {"channel_id": "7", "value": "quiet", "preview": False},
                self.context,
            )

        plan = self.gateway.create_plan(
            "apply_setting",
            {"channel_id": "7", "value": "quiet", "preview": False},
            self.context,
        )
        approval = self.gateway.approve(plan.plan_id, self.context)
        self.assertNotEqual(approval.approval_id, plan.plan_id)

        result = self.gateway.execute(
            "apply_setting",
            None,
            self.context,
            approval_id=approval.approval_id,
        )

        self.assertEqual(result["arguments"]["preview"], False)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(ApprovalConsumedError):
            self.gateway.execute(
                "apply_setting",
                None,
                self.context,
                approval_id=approval.approval_id,
            )

        context = TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=self.actor_id,
        )
        with self.pool.transaction(context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT
                    encode(approval_token_hash, 'hex'),
                    status,
                    octet_length(approval_token_hash),
                    approval_token_hash::text
                FROM agent.action_approvals
                WHERE tenant_id = %s
                """,
                (self.tenant_id,),
            ).fetchone()
            plan_status = connection.execute(
                """
                SELECT status
                FROM agent.action_plans
                WHERE tenant_id = %s AND id = %s
                """,
                (self.tenant_id, plan.plan_id),
            ).fetchone()[0]
        self.assertEqual(row[1], "consumed")
        self.assertEqual(row[2], 32)
        self.assertNotIn(approval.approval_id, row[0])
        self.assertNotIn(approval.approval_id, row[3])
        self.assertEqual(plan_status, "executed")

    def test_approval_is_bound_to_actor_tenant_action_and_arguments(self):
        plan = self.gateway.create_plan(
            "apply_setting",
            {"channel_id": "7", "value": "quiet", "preview": False},
            self.context,
        )
        approval = self.gateway.approve(plan.plan_id, self.context)
        other_context = ToolExecutionContext(
            actor_id=self.other_actor_id,
            tenant_id=self.tenant_id,
            permissions={"probe.write"},
            allowed_channel_ids={"7"},
            agent_session_id=str(uuid4()),
        )

        with self.assertRaises(ApprovalError):
            self.gateway.execute(
                "apply_setting",
                {"channel_id": "7", "value": "loud", "preview": False},
                self.context,
                approval_id=approval.approval_id,
            )
        with self.assertRaises(ApprovalError):
            self.gateway.execute(
                "apply_setting",
                None,
                other_context,
                approval_id=approval.approval_id,
            )
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
