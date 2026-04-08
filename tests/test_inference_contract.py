from __future__ import annotations

import pytest

import inference
from ecom.models import EcomAction
from inference import (
    _build_llm_client,
    _extract_last_action_error,
    _enforce_action_contract,
    _probe_llm_proxy,
    log_end,
)


class _Obs:
    def __init__(self, info):
        self.info = info


def test_extract_last_action_error_reads_invalid_action_fallback() -> None:
    obs = _Obs(info={"invalid_action": "invalid_final_action"})

    value = _extract_last_action_error(obs)

    assert value == "invalid_final_action"


def test_enforce_action_contract_rejects_action_not_in_available_actions() -> None:
    obs = _Obs(info={"available_actions": ["APPROVE", "ESCALATE"]})

    action = EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")
    constrained = _enforce_action_contract(obs, action)

    assert constrained is None


def test_enforce_action_contract_rejects_unknown_reject_reason() -> None:
    obs = _Obs(
        info={
            "available_actions": ["REJECT"],
            "reject_reason_codes": ["TIME_EXPIRED"],
        }
    )

    action = EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")
    constrained = _enforce_action_contract(obs, action)

    assert constrained is None


def test_enforce_action_contract_allows_valid_reject_reason() -> None:
    obs = _Obs(
        info={
            "available_actions": ["REJECT"],
            "reject_reason_codes": ["SUSPECTED_FRAUD"],
        }
    )

    action = EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")
    constrained = _enforce_action_contract(obs, action)

    assert constrained is not None
    assert constrained.action_type == "REJECT"
    assert constrained.reason_code == "SUSPECTED_FRAUD"


def test_build_llm_client_uses_injected_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class DummyOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(inference, "OpenAI", DummyOpenAI)
    monkeypatch.setenv("HF_TOKEN", "hf-proxy-token")
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")

    client = _build_llm_client()

    assert isinstance(client, DummyOpenAI)
    assert captured["api_key"] == "hf-proxy-token"
    assert captured["base_url"] == "https://proxy.example/v1"


def test_build_llm_client_fails_without_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(inference, "OpenAI", DummyOpenAI)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _build_llm_client()


def test_build_llm_client_uses_default_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class DummyOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(inference, "OpenAI", DummyOpenAI)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-proxy-token")
    monkeypatch.delenv("API_BASE_URL", raising=False)

    client = _build_llm_client()

    assert isinstance(client, DummyOpenAI)
    assert captured["api_key"] == "hf-proxy-token"
    assert captured["base_url"] == "https://api.openai.com/v1"


def test_model_name_reads_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_NAME", "gpt-test-model")

    assert inference._model_name() == "gpt-test-model"


def test_probe_llm_proxy_makes_chat_completion_request() -> None:
    captured = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()

    _probe_llm_proxy(DummyClient())

    assert captured["model"] == inference._model_name()
    assert captured["stream"] is False
    assert captured["max_tokens"] == 2


def test_inference_runner_allows_timeout_recovery_step() -> None:
    assert inference.MAX_STEPS == 5


def test_log_end_uses_required_format_without_score(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_end(success=True, steps=2, rewards=[0.08, 0.65])
    out = capsys.readouterr().out.strip()

    assert out.startswith("[END] ")
    assert "success=true" in out
    assert "steps=2" in out
    assert "score=" not in out
    assert out.endswith("rewards=0.08,0.65")
