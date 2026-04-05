from __future__ import annotations

from ecom.models import EcomAction
from inference import (
    _extract_last_action_error,
    _enforce_action_contract,
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
