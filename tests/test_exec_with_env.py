from __future__ import annotations

import os

import pytest

from scripts import exec_with_env


def test_configured_environment_overrides_snapshot_without_shell_evaluation(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text(
        "EVOSSEARCH_PORT='5000'\n"
        'EVOSSEARCH_LABEL="literal $HOME; $(touch /tmp/not-run)"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOSSEARCH_PORT", "wrong")

    environment = exec_with_env.configured_environment(env_file)

    assert environment["EVOSSEARCH_PORT"] == "5000"
    assert environment["EVOSSEARCH_LABEL"] == "literal $HOME; $(touch /tmp/not-run)"
    assert environment["EVOSSEARCH_CONFIG_ENV_FILE"] == str(env_file)


def test_configured_environment_rejects_invalid_variable_name(tmp_path):
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text("BAD-NAME=value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid variable names"):
        exec_with_env.configured_environment(env_file)


def test_main_execs_requested_command_with_canonical_values(tmp_path, monkeypatch):
    env_file = tmp_path / "eva-ai.env"
    env_file.write_text("EVOSSEARCH_PORT=5000\n", encoding="utf-8")
    observed = {}

    def fake_execvpe(executable, command, environment):
        observed.update(
            executable=executable,
            command=command,
            environment=environment,
        )
        raise RuntimeError("exec boundary")

    monkeypatch.setattr(os, "execvpe", fake_execvpe)
    with pytest.raises(RuntimeError, match="exec boundary"):
        exec_with_env.main(
            ["--env-file", str(env_file), "--", "/bin/echo", "ready"]
        )

    assert observed["executable"] == "/bin/echo"
    assert observed["command"] == ["/bin/echo", "ready"]
    assert observed["environment"]["EVOSSEARCH_PORT"] == "5000"
