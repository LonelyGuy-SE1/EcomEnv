from __future__ import annotations

from ecom.models import EcomAction
from ecom.server.ecom_environment import EcomEnvironment


def _new_env() -> EcomEnvironment:
    return EcomEnvironment(mode="medium")


def test_step_before_reset_ignores_action_and_exposes_error_code() -> None:
    env = _new_env()

    obs = env.step(EcomAction(action_type="APPROVE"))

    assert obs.done is False
    assert obs.reward is None
    assert obs.info.get("invalid_action") == "step_called_before_reset_action_ignored"
    assert (
        obs.info.get("last_action_error") == "step_called_before_reset_action_ignored"
    )
    assert "APPROVE" in obs.info.get("available_actions", [])


def test_step_after_done_returns_terminal_observation_not_exception() -> None:
    env = _new_env()

    env.reset(seed=222)
    terminal = env.step(EcomAction(action_type="APPROVE"))
    assert terminal.done is True

    post_terminal = env.step(EcomAction(action_type="APPROVE"))
    assert post_terminal.done is True
    assert post_terminal.reward == 0.0
    assert (
        post_terminal.info.get("invalid_action")
        == "episode_already_terminated_call_reset"
    )
    assert (
        post_terminal.info.get("last_action_error")
        == "episode_already_terminated_call_reset"
    )
    assert post_terminal.info.get("available_actions") == []


def test_repeated_request_info_is_penalized_with_machine_readable_error() -> None:
    env = _new_env()

    env.reset(seed=222)
    first = env.step(EcomAction(action_type="REQUEST_INFO"))
    assert first.done is False

    second = env.step(EcomAction(action_type="REQUEST_INFO"))
    assert second.done is False
    assert second.reward == -0.10
    assert second.info.get("invalid_action") == "request_info_already_used"
    assert second.info.get("last_action_error") == "request_info_already_used"
    assert second.info.get("available_actions") == ["APPROVE", "REJECT", "ESCALATE"]


def test_terminal_reward_is_bounded_and_grader_fields_exist() -> None:
    env = _new_env()

    env.reset(seed=111)
    obs = env.step(EcomAction(action_type="APPROVE"))

    assert obs.done is True
    assert isinstance(obs.reward, float)
    assert 0.0 <= obs.reward <= 1.0
    assert isinstance(obs.info.get("grader_score"), float)
    assert isinstance(obs.info.get("grader_success"), bool)


def test_task_seed_is_deterministic_for_same_task_name() -> None:
    env_a = EcomEnvironment(task_name="medium_balanced_judgment")
    env_b = EcomEnvironment(task_name="medium_balanced_judgment")

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    assert obs_a.return_reason == obs_b.return_reason
    assert obs_a.product_category == obs_b.product_category
    assert obs_a.product_value == obs_b.product_value
    assert obs_a.days_since_purchase == obs_b.days_since_purchase
    assert obs_a.user_account_age_days == obs_b.user_account_age_days
    assert abs(obs_a.return_rate - obs_b.return_rate) < 1e-12


def test_task_grader_success_thresholds_match_spec() -> None:
    expected = {
        "easy_policy_compliance": 0.75,
        "medium_balanced_judgment": 0.68,
        "hard_conflicting_signals": 0.74,
    }

    for task_name, threshold in expected.items():
        env = EcomEnvironment(task_name=task_name)
        env.reset()
        obs = env.step(EcomAction(action_type="APPROVE"))
        score = float(obs.info["grader_score"])
        success = bool(obs.info["grader_success"])
        assert success == (score >= threshold)


def test_terminal_info_contains_decision_audit_with_counterfactuals() -> None:
    env = EcomEnvironment(task_name="medium_balanced_judgment")

    env.reset()
    obs = env.step(EcomAction(action_type="APPROVE"))

    assert obs.done is True
    audit = obs.info.get("decision_audit")
    assert isinstance(audit, dict)
    assert isinstance(audit.get("chosen_action"), str)
    assert isinstance(audit.get("chosen_reward"), float)
    assert isinstance(audit.get("best_counterfactual_reward"), float)
    assert isinstance(audit.get("decision_gap"), float)
    assert 0.0 <= float(audit.get("chosen_reward")) <= 1.0
    assert 0.0 <= float(audit.get("best_counterfactual_reward")) <= 1.0
    assert 0.0 <= float(audit.get("decision_gap")) <= 1.0

    counterfactual_rewards = audit.get("counterfactual_rewards")
    assert isinstance(counterfactual_rewards, dict)
    assert "APPROVE" in counterfactual_rewards
    assert "ESCALATE" in counterfactual_rewards
    assert "REJECT(TIME_EXPIRED)" in counterfactual_rewards
    assert "REJECT(POLICY_VIOLATION)" in counterfactual_rewards
    assert "REJECT(SUSPECTED_FRAUD)" in counterfactual_rewards

    policy_flags = audit.get("policy_flags")
    assert isinstance(policy_flags, dict)
    assert isinstance(policy_flags.get("time_policy_violated"), bool)
    assert isinstance(policy_flags.get("category_policy_violated"), bool)
    assert isinstance(policy_flags.get("exception_applies"), bool)
    assert isinstance(policy_flags.get("ambiguous_case"), bool)


def test_timeout_path_contains_decision_audit() -> None:
    env = EcomEnvironment(task_name="easy_policy_compliance")

    env.reset()
    env.step(EcomAction(action_type="REQUEST_INFO"))
    env.step(EcomAction(action_type="REQUEST_INFO"))
    env.step(EcomAction(action_type="REQUEST_INFO"))
    env.step(EcomAction(action_type="REQUEST_INFO"))
    timeout_obs = env.step(EcomAction(action_type="REQUEST_INFO"))

    assert timeout_obs.done is True
    assert timeout_obs.info.get("termination_reason") == "max_steps_exceeded"
    audit = timeout_obs.info.get("decision_audit")
    assert isinstance(audit, dict)
    assert audit.get("chosen_action") == "NONE"
    assert isinstance(audit.get("counterfactual_rewards"), dict)
