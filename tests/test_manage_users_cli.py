from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts import manage_users
from security import ALL_CHANNELS, Role


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


if __name__ == "__main__":
    unittest.main()
