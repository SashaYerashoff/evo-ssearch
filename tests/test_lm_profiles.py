import os
import unittest
from unittest.mock import patch

import oldapp
from config import _get_lm_profiles


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class LmProfileConfigTests(unittest.TestCase):
    def test_config_parser_adds_named_profiles(self):
        env = {
            "EVOSSEARCH_LM_PROFILES": "agent,vlm-east",
            "EVOSSEARCH_LM_PROFILE_AGENT_KIND": "agent",
            "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL": "http://agent.local/v1/",
            "EVOSSEARCH_LM_PROFILE_AGENT_MODEL": "qwen3.6-27b",
            "EVOSSEARCH_LM_PROFILE_AGENT_API_KEY": "agent-secret",
            "EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT": "600",
            "EVOSSEARCH_LM_PROFILE_VLM_EAST_KIND": "vlm",
            "EVOSSEARCH_LM_PROFILE_VLM_EAST_BASE_URL": "http://vlm-east.local/v1",
            "EVOSSEARCH_LM_PROFILE_VLM_EAST_MODEL": "qwen3.5-vl-4b",
        }
        with patch.dict(os.environ, env, clear=True):
            profiles = _get_lm_profiles(
                base_url="http://default.local/v1",
                model="default-model",
                api_key="default-secret",
                timeout=120,
            )

        self.assertEqual(profiles["default"]["base_url"], "http://default.local/v1")
        self.assertEqual(profiles["agent"]["kind"], "agent")
        self.assertEqual(profiles["agent"]["base_url"], "http://agent.local/v1")
        self.assertEqual(profiles["agent"]["model"], "qwen3.6-27b")
        self.assertEqual(profiles["agent"]["api_key"], "agent-secret")
        self.assertEqual(profiles["agent"]["timeout"], 600)
        self.assertEqual(profiles["vlm-east"]["kind"], "vlm")
        self.assertEqual(profiles["vlm-east"]["model"], "qwen3.5-vl-4b")


class LmProfileRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._orig_cache_payload = oldapp._lm_models_cache_payload
        self._orig_cache_expires = oldapp._lm_models_cache_expires_at
        oldapp._lm_models_cache_payload = None
        oldapp._lm_models_cache_expires_at = 0.0

    def tearDown(self):
        oldapp._lm_models_cache_payload = self._orig_cache_payload
        oldapp._lm_models_cache_expires_at = self._orig_cache_expires

    def test_chat_routes_model_selector_to_profile_endpoint(self):
        profiles = {
            "default": {
                "id": "default",
                "kind": "general",
                "base_url": "http://default.local/v1",
                "model": "default-model",
                "api_key": "",
                "timeout": 120,
            },
            "vlm-a": {
                "id": "vlm-a",
                "kind": "vlm",
                "base_url": "http://vlm-a.local/v1",
                "model": "qwen3.5-vl-4b",
                "api_key": "vlm-secret",
                "timeout": 321,
            },
        }
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            captured["json"] = kwargs.get("json")
            captured["timeout"] = kwargs.get("timeout")
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        with (
            patch.object(oldapp.config, "LM_PROFILES", profiles),
            patch.object(oldapp.config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(oldapp.requests, "post", fake_post),
        ):
            result = oldapp._call_lm_chat(
                [{"role": "user", "content": "describe"}],
                model_override="vlm-a",
                profile_kind="vlm",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["url"], "http://vlm-a.local/v1/chat/completions")
        self.assertEqual(captured["json"]["model"], "qwen3.5-vl-4b")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer vlm-secret")
        self.assertEqual(captured["timeout"], 321)

    def test_model_catalog_exposes_profiles_without_api_keys(self):
        profiles = {
            "default": {
                "id": "default",
                "kind": "general",
                "base_url": "http://default.local/v1",
                "model": "default-model",
                "api_key": "",
                "timeout": 120,
            },
            "agent": {
                "id": "agent",
                "kind": "agent",
                "base_url": "http://agent.local/v1",
                "model": "qwen-agent",
                "api_key": "agent-secret",
                "timeout": 600,
            },
            "vlm-a": {
                "id": "vlm-a",
                "kind": "vlm",
                "base_url": "http://vlm-a.local/v1",
                "model": "qwen-vlm",
                "api_key": "vlm-secret",
                "timeout": 300,
            },
        }

        def fake_get(url, **_kwargs):
            if "agent" in url:
                return _Response({"data": [{"id": "qwen-agent"}]})
            return _Response({"data": [{"id": "qwen-vlm"}]})

        with (
            patch.object(oldapp.config, "LM_PROFILES", profiles),
            patch.object(oldapp.config, "LM_AGENT_PROFILE_ID", "agent"),
            patch.object(oldapp.config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(oldapp.requests, "get", fake_get),
        ):
            payload = oldapp._fetch_lm_model_catalog(force=True)

        self.assertEqual(payload["default_model"], "vlm-a")
        self.assertIn("agent", payload["models"])
        self.assertIn("vlm-a", payload["models"])
        self.assertIn("qwen-agent", payload["models"])
        self.assertIn("qwen-vlm", payload["models"])
        self.assertNotIn("agent-secret", str(payload))
        self.assertNotIn("vlm-secret", str(payload))


if __name__ == "__main__":
    unittest.main()
