import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import oldapp
from config import _get_lm_profiles


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def close(self):
        return None


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
            "EVOSSEARCH_LM_PROFILE_VLM_EAST_ENABLED": "false",
            "EVOSSEARCH_LM_PROFILE_VLM_EAST_GPU": "server-a:0",
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
        self.assertFalse(profiles["vlm-east"]["enabled"])
        self.assertEqual(profiles["vlm-east"]["gpu"], "server-a:0")


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

        class AdmissionCapture:
            def admission(self, _resource, *, workload, **_kwargs):
                captured["workload"] = workload
                return nullcontext()

        with (
            patch.object(oldapp.config, "LM_PROFILES", profiles),
            patch.object(oldapp.config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(oldapp.requests, "post", fake_post),
            patch.object(oldapp, "_lm_admission_controller", AdmissionCapture()),
        ):
            result = oldapp._call_lm_chat(
                [{"role": "user", "content": "describe"}],
                model_override="vlm-a",
                profile_kind="vlm",
                workload_class="rollup",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["url"], "http://vlm-a.local/v1/chat/completions")
        self.assertEqual(captured["json"]["model"], "qwen3.5-vl-4b")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer vlm-secret")
        self.assertEqual(captured["timeout"], 321)
        self.assertEqual(captured["workload"], "rollup")
        self.assertEqual(
            captured["json"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_interactive_agent_keeps_model_thinking_mode(self):
        profile = {
            "id": "agent",
            "kind": "agent",
            "base_url": "http://agent.local/v1",
            "model": "qwen3.5-9b-mtp",
            "api_key": "",
            "timeout": 120,
        }
        captured = {}

        def fake_post(_url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        class AdmissionCapture:
            def admission(self, _resource, **_kwargs):
                return nullcontext()

        with (
            patch.object(oldapp, "_resolve_lm_profile", return_value=profile),
            patch.object(oldapp.requests, "post", fake_post),
            patch.object(oldapp, "_lm_admission_controller", AdmissionCapture()),
        ):
            result = oldapp._call_lm_chat(
                [{"role": "user", "content": "research this"}],
                profile_id="agent",
                profile_kind="agent",
                workload_class="agent",
            )

        self.assertEqual(result, "ok")
        self.assertNotIn("chat_template_kwargs", captured["json"])

    def test_interactive_vlm_requests_direct_answer(self):
        profile = {
            "id": "vlm",
            "kind": "vlm",
            "base_url": "http://vlm.local/v1",
            "model": "qwen3.5-4b-mtp",
            "api_key": "",
            "timeout": 120,
        }
        captured = {}

        def fake_post(_url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        class AdmissionCapture:
            def admission(self, _resource, **_kwargs):
                return nullcontext()

        with (
            patch.object(oldapp, "_resolve_lm_profile", return_value=profile),
            patch.object(oldapp.requests, "post", fake_post),
            patch.object(oldapp, "_lm_admission_controller", AdmissionCapture()),
        ):
            result = oldapp._call_lm_chat(
                [{"role": "user", "content": "describe this frame"}],
                profile_id="vlm",
                profile_kind="vlm",
                workload_class="describe",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            captured["json"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

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
                "gpu": "infer-a:0",
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
            patch.object(oldapp.config, "LM_VLM_BALANCER_ENABLED", True),
            patch.object(oldapp.config, "LM_VLM_BALANCER_PROFILES", ("vlm-a",)),
            patch.object(oldapp.requests, "get", fake_get),
        ):
            payload = oldapp._fetch_lm_model_catalog(force=True)

        self.assertEqual(payload["default_model"], "vlm-a")
        self.assertEqual(payload["offline_default_model"], "agent")
        self.assertEqual(payload["agent_default_model"], "agent")
        self.assertEqual(sorted(payload["models"]), ["qwen-agent", "qwen-vlm"])
        self.assertNotIn("agent", payload["models"])
        self.assertNotIn("vlm-a", payload["models"])
        self.assertIn("agent", payload["configured_models"])
        self.assertIn("vlm-a", payload["configured_models"])
        agent_profile = next(profile for profile in payload["profiles"] if profile["id"] == "agent")
        self.assertTrue(agent_profile["available"])
        self.assertEqual(agent_profile["available_models"], ["qwen-agent"])
        self.assertEqual(payload["auto_model_selector"], "__auto__")
        self.assertTrue(payload["vlm_balancer"]["enabled"])
        self.assertEqual(payload["vlm_balancer"]["profile_ids"], ["vlm-a"])
        public_vlm = next(profile for profile in payload["profiles"] if profile["id"] == "vlm-a")
        self.assertEqual(public_vlm["gpu"], "infer-a:0")
        self.assertNotIn("agent-secret", str(payload))
        self.assertNotIn("vlm-secret", str(payload))

    def test_auto_balancer_assigns_stable_vlm_profile(self):
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
                "model": "qwen-vlm",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
            "vlm-b": {
                "id": "vlm-b",
                "kind": "vlm",
                "base_url": "http://vlm-b.local/v1",
                "model": "qwen-vlm",
                "api_key": "",
                "timeout": 300,
                "enabled": True,
            },
            "vlm-disabled": {
                "id": "vlm-disabled",
                "kind": "vlm",
                "base_url": "http://disabled.local/v1",
                "model": "qwen-vlm",
                "api_key": "",
                "timeout": 300,
                "enabled": False,
            },
        }
        with (
            patch.object(oldapp.config, "LM_PROFILES", profiles),
            patch.object(oldapp.config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(oldapp.config, "LM_VLM_BALANCER_ENABLED", True),
            patch.object(
                oldapp.config,
                "LM_VLM_BALANCER_PROFILES",
                ("vlm-a", "vlm-b", "vlm-disabled"),
            ),
        ):
            first_hint, first_meta = oldapp._resolve_luxriot_vlm_model_hint(7, "__auto__")
            second_hint, second_meta = oldapp._resolve_luxriot_vlm_model_hint(7, "")
            manual_hint, manual_meta = oldapp._resolve_luxriot_vlm_model_hint(7, "vlm-b")

        self.assertIn(first_hint, {"vlm-a", "vlm-b"})
        self.assertEqual(second_hint, first_hint)
        self.assertEqual(first_meta["mode"], "auto")
        self.assertEqual(second_meta["assigned_profile_id"], first_hint)
        self.assertEqual(first_meta["profile_count"], 2)
        self.assertEqual(manual_hint, "vlm-b")
        self.assertEqual(manual_meta["mode"], "manual")

    def test_ready_checks_required_vlm_profiles_without_secrets(self):
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
                "model": "qwen-vlm",
                "api_key": "vlm-secret",
                "timeout": 300,
                "enabled": True,
                "gpu": "server-a:0",
            },
        }
        captured_headers = []

        def fake_get(_url, **kwargs):
            captured_headers.append(kwargs.get("headers") or {})
            return _Response({"data": [{"id": "qwen-vlm"}]})

        with (
            patch.object(oldapp.config, "LM_PROFILES", profiles),
            patch.object(oldapp.config, "LM_VLM_PROFILE_ID", "vlm-a"),
            patch.object(oldapp.config, "LM_VLM_BALANCER_ENABLED", True),
            patch.object(oldapp.config, "LM_VLM_BALANCER_PROFILES", ("vlm-a",)),
            patch.object(oldapp.requests, "get", fake_get),
        ):
            payload = oldapp._check_lm_profiles_ready(timeout_sec=0.1)

        self.assertTrue(payload["ok"], payload)
        self.assertTrue(payload["required"])
        self.assertEqual(payload["required_profile_ids"], ["vlm-a"])
        public_vlm = next(profile for profile in payload["profiles"] if profile["id"] == "vlm-a")
        self.assertEqual(public_vlm["gpu"], "server-a:0")
        self.assertIn("Bearer vlm-secret", str(captured_headers))
        self.assertNotIn("vlm-secret", str(payload))


if __name__ == "__main__":
    unittest.main()
