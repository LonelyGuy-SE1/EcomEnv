from __future__ import annotations

import pytest

import inference
from ecom.models import EcomAction
from inference import (
    _build_llm_client,
    _extract_last_action_error,
    _enforce_action_contract,
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
    monkeypatch.setenv("API_KEY", "proxy-token")
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")

    client = _build_llm_client()

    assert isinstance(client, DummyOpenAI)
    assert captured["api_key"] == "proxy-token"
    assert captured["base_url"] == "https://proxy.example/v1"


def test_build_llm_client_fails_without_required_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(inference, "OpenAI", DummyOpenAI)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")

    with pytest.raises(RuntimeError, match="API_KEY"):
        _build_llm_client()


def test_log_end_includes_score_field(capsys: pytest.CaptureFixture[str]) -> None:
    log_end(success=True, steps=2, score=0.73, rewards=[0.08, 0.65])
    out = capsys.readouterr().out.strip()

    assert out.startswith("[END] ")
    assert "success=true" in out
    assert "steps=2" in out
    assert "score=0.73" in out
    assert out.endswith("rewards=0.08,0.65")
