from unittest.mock import patch

import oldapp


def test_runtime_services_start_only_from_explicit_idempotent_bootstrap():
    original_initialized = oldapp._runtime_services_initialized
    original_embedder = dict(oldapp._runtime_embedder_result)
    original_restore = dict(oldapp._luxriot_restore_result)
    original_eager = oldapp.config.EMBEDDER_EAGER_LOAD
    try:
        oldapp._runtime_services_initialized = False
        oldapp._runtime_embedder_result = {
            "ok": False,
            "status": "not_initialized",
        }
        oldapp._luxriot_restore_result = {
            "ok": False,
            "status": "not_initialized",
        }
        oldapp.config.EMBEDDER_EAGER_LOAD = True
        with (
            patch("oldapp.ensure_embedder_loaded") as load_embedder,
            patch("oldapp._configure_inference_queue") as configure_queue,
            patch.object(
                oldapp.luxriot_manager,
                "restore_desired_live_sessions",
                return_value={"ok": True, "status": "restored"},
            ) as restore,
        ):
            oldapp.initialize_runtime_services()
            oldapp.initialize_runtime_services()

        load_embedder.assert_called_once_with(oldapp.active_embedder)
        configure_queue.assert_called_once_with()
        restore.assert_called_once_with()
        assert oldapp._luxriot_restore_result["status"] == "restored"
    finally:
        oldapp._runtime_services_initialized = original_initialized
        oldapp._runtime_embedder_result = original_embedder
        oldapp._luxriot_restore_result = original_restore
        oldapp.config.EMBEDDER_EAGER_LOAD = original_eager


def test_embedder_bootstrap_failure_stays_live_unready_and_starts_no_runtime():
    original_initialized = oldapp._runtime_services_initialized
    original_embedder = dict(oldapp._runtime_embedder_result)
    original_restore = dict(oldapp._luxriot_restore_result)
    original_eager = oldapp.config.EMBEDDER_EAGER_LOAD
    original_secure = oldapp.config.SECURE_DEPLOYMENT_REQUIRED
    original_queue_enabled = oldapp.config.INFERENCE_QUEUE_ENABLED
    try:
        oldapp._runtime_services_initialized = False
        oldapp._runtime_embedder_result = {
            "ok": False,
            "status": "not_initialized",
        }
        oldapp._luxriot_restore_result = {
            "ok": False,
            "status": "not_initialized",
        }
        oldapp.config.EMBEDDER_EAGER_LOAD = True
        oldapp.config.SECURE_DEPLOYMENT_REQUIRED = False
        oldapp.config.INFERENCE_QUEUE_ENABLED = True
        with (
            patch(
                "oldapp.ensure_embedder_loaded",
                side_effect=RuntimeError("CUDA model unavailable"),
            ) as load_embedder,
            patch("oldapp._configure_inference_queue") as configure_queue,
            patch.object(
                oldapp.luxriot_manager,
                "restore_desired_live_sessions",
            ) as restore,
        ):
            oldapp.initialize_runtime_services()
            oldapp.initialize_runtime_services()

        load_embedder.assert_called_once_with(oldapp.active_embedder)
        configure_queue.assert_not_called()
        restore.assert_not_called()
        assert oldapp._runtime_services_initialized is True
        assert oldapp._runtime_embedder_result["status"] == "load_failed"
        assert oldapp._runtime_embedder_result["error"] == "RuntimeError"
        assert oldapp._luxriot_restore_result["status"] == "blocked_embedder"
        assert oldapp._runtime_capture_bootstrap_allowed() is False

        ready_component = {"ok": True, "status": "ready", "required": False}
        required_component = {"ok": True, "status": "ready", "required": True}
        with (
            patch("oldapp._check_database_ready", return_value=required_component),
            patch("oldapp._check_postgres_ready", return_value=ready_component),
            patch("oldapp._check_auth_ready", return_value=ready_component),
            patch("oldapp._check_deployment_security_ready", return_value=ready_component),
            patch("oldapp._check_attention_ready", return_value=ready_component),
            patch("oldapp._check_lm_profiles_ready", return_value=ready_component),
            patch("oldapp._check_luxriot_ready", return_value=ready_component),
            patch("oldapp._configure_inference_queue") as readiness_queue,
        ):
            response = oldapp.app.test_client().get("/ready")

        assert response.status_code == 503
        payload = response.get_json()
        assert payload["checks"]["embedder"]["status"] == "load_failed"
        assert payload["checks"]["embedder"]["error"] == "RuntimeError"
        assert payload["checks"]["inference_queue"]["status"] == "blocked_embedder"
        assert payload["checks"]["luxriot_restore"]["status"] == "blocked_embedder"
        readiness_queue.assert_not_called()
    finally:
        oldapp._runtime_services_initialized = original_initialized
        oldapp._runtime_embedder_result = original_embedder
        oldapp._luxriot_restore_result = original_restore
        oldapp.config.EMBEDDER_EAGER_LOAD = original_eager
        oldapp.config.SECURE_DEPLOYMENT_REQUIRED = original_secure
        oldapp.config.INFERENCE_QUEUE_ENABLED = original_queue_enabled
