"""Baseline inference runner for the Ecom returns decision environment."""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ecom import EcomAction, EcomEnv

BENCHMARK = "ecom_returns_decision"
MAX_STEPS = 5
TEMPERATURE = 0
MAX_TOKENS = 180

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a returns operations agent.
    Choose exactly one action in JSON only.

    Allowed action_type values:
    - APPROVE
    - REJECT
    - ESCALATE
    - REQUEST_INFO

    If action_type is REJECT, include reason_code with one of:
    - TIME_EXPIRED
    - POLICY_VIOLATION
    - SUSPECTED_FRAUD

    Output JSON only. No prose, no markdown.
    """
).strip()


def _model_name() -> str:
    return os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _image_name() -> Optional[str]:
    return os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")


def _env_base_url() -> Optional[str]:
    return os.getenv("ENV_BASE_URL")


def _task_names() -> List[str]:
    task_name = os.getenv("ECOM_TASK_NAME") or os.getenv("ECOM_TASK")
    if task_name:
        return [task_name]
    return [
        "easy_policy_compliance",
        "medium_balanced_judgment",
        "hard_conflicting_signals",
    ]


@dataclass
class EpisodeOutcome:
    success: bool
    steps: int
    score: float
    rewards: List[float]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def format_action(action: EcomAction) -> str:
    if action.reason_code is None:
        return action.action_type
    return f"{action.action_type}({action.reason_code})"


def extract_return_window(policy_summary: str) -> int:
    match = re.search(r"within\s+(\d+)\s+days", policy_summary, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 30


def exception_applies(observation: Any) -> bool:
    reason = str(observation.return_reason).lower()
    policy_summary = str(observation.policy_summary).lower()

    match = re.search(r"exception:\s*([^.]*)", policy_summary)
    clause = match.group(1) if match else ""

    if reason == "damaged-shipping" and (
        "damage in transit" in clause or "damaged" in clause
    ):
        return True

    if reason == "defective" and "defective" in clause:
        return True

    return False


def is_restricted_class_case(observation: Any) -> bool:
    return "restricted class" in str(observation.product_condition_notes).lower()


def should_reject_time_expired(
    observation: Any, window: int, has_exception: bool
) -> bool:
    if observation.days_since_purchase <= window:
        return False
    if has_exception and not is_restricted_class_case(observation):
        return False
    return True


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def _extract_last_action_error(observation: Any) -> Optional[str]:
    info = getattr(observation, "info", None)
    if not isinstance(info, dict):
        return None

    for key in ("last_action_error", "invalid_action"):
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_available_actions(observation: Any) -> List[str]:
    info = getattr(observation, "info", None)
    if not isinstance(info, dict):
        return []

    raw = info.get("available_actions")
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw]


def _extract_reject_reason_codes(observation: Any) -> List[str]:
    info = getattr(observation, "info", None)
    if not isinstance(info, dict):
        return []

    raw = info.get("reject_reason_codes")
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw]


def _enforce_action_contract(
    observation: Any, action: EcomAction
) -> Optional[EcomAction]:
    available_actions = _extract_available_actions(observation)
    if available_actions and action.action_type not in set(available_actions):
        return None

    if action.action_type == "REJECT":
        valid_reasons = set(_extract_reject_reason_codes(observation))
        if valid_reasons and action.reason_code not in valid_reasons:
            return None

    return action


def heuristic_policy(observation: Any, step: int) -> EcomAction:
    available_actions = set(_extract_available_actions(observation))

    window = extract_return_window(observation.policy_summary)
    has_exception = exception_applies(observation)
    notes = str(observation.product_condition_notes).lower()
    reason = str(observation.return_reason)
    return_rate = float(observation.return_rate)

    ambiguous = (
        ("mixed indicators" in notes)
        or ("conflict" in notes)
        or (0.40 <= return_rate <= 0.65)
        or (observation.days_since_purchase > window and has_exception)
    )
    if step == 1 and (not available_actions or "REQUEST_INFO" in available_actions):
        if ambiguous:
            return EcomAction(action_type="REQUEST_INFO")

    if should_reject_time_expired(observation, window, has_exception):
        if available_actions and "REJECT" not in available_actions:
            return EcomAction(action_type="APPROVE")
        return EcomAction(action_type="REJECT", reason_code="TIME_EXPIRED")

    if "restricted class" in notes:
        if available_actions and "REJECT" not in available_actions:
            return EcomAction(action_type="APPROVE")
        return EcomAction(action_type="REJECT", reason_code="POLICY_VIOLATION")

    if (
        step >= 2
        and observation.product_value == "high"
        and return_rate >= 0.50
        and (
            "conflict" in notes
            or "disputed evidence" in notes
            or reason in ("changed-mind", "wrong-item")
        )
    ):
        if available_actions and "REJECT" not in available_actions:
            return EcomAction(action_type="APPROVE")
        return EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")

    if return_rate >= 0.60 and observation.product_value == "high":
        if available_actions and "REJECT" not in available_actions:
            return EcomAction(action_type="APPROVE")
        return EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")

    if reason in ("defective", "wrong-item", "damaged-shipping") and return_rate < 0.55:
        if available_actions and "APPROVE" not in available_actions:
            return EcomAction(action_type="ESCALATE")
        return EcomAction(action_type="APPROVE")

    if return_rate >= 0.55:
        if available_actions and "ESCALATE" not in available_actions:
            return EcomAction(action_type="APPROVE")
        return EcomAction(action_type="ESCALATE")

    if available_actions and "APPROVE" not in available_actions:
        if "ESCALATE" in available_actions:
            return EcomAction(action_type="ESCALATE")
        if "REJECT" in available_actions:
            return EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")

    return EcomAction(action_type="APPROVE")


def build_user_prompt(step: int, observation: Any, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    available_actions = _extract_available_actions(observation)
    reject_reason_codes = _extract_reject_reason_codes(observation)

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        return_reason: {observation.return_reason}
        product_category: {observation.product_category}
        product_value: {observation.product_value}
        days_since_purchase: {observation.days_since_purchase}
        user_account_age_days: {observation.user_account_age_days}
        product_condition_notes: {observation.product_condition_notes}
        return_rate: {float(observation.return_rate):.3f}
        total_orders: {observation.total_orders}
        policy_summary: {observation.policy_summary}
        available_actions: {", ".join(available_actions) if available_actions else "None"}
        available_reject_reason_codes: {", ".join(reject_reason_codes) if reject_reason_codes else "None"}
        Previous steps:
        {history_block}
        """
    ).strip()

    return prompt


def get_model_action(
    client: OpenAI, step: int, observation: Any, history: List[str]
) -> Optional[EcomAction]:
    user_prompt = build_user_prompt(step, observation, history)

    try:
        completion = client.chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        return None

    data = _safe_json_parse(text)
    if data is None:
        return None

    action_type = str(data.get("action_type", "")).strip().upper()
    reason_code = data.get("reason_code")
    if reason_code is not None:
        reason_code = str(reason_code).strip().upper()

    if action_type == "REJECT":
        if reason_code not in {
            "TIME_EXPIRED",
            "POLICY_VIOLATION",
            "SUSPECTED_FRAUD",
        }:
            return None
        action = EcomAction(action_type="REJECT", reason_code=reason_code)
        return _enforce_action_contract(observation, action)

    if action_type in {"APPROVE", "ESCALATE", "REQUEST_INFO"}:
        action = EcomAction(action_type=action_type)
        return _enforce_action_contract(observation, action)

    return None


def _build_llm_client() -> OpenAI:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN environment variable is required.")
    return OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )


def _probe_llm_proxy(client: OpenAI) -> None:
    try:
        client.chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "OK"},
            ],
            temperature=0,
            max_tokens=2,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to make an LLM request through API_BASE_URL with HF_TOKEN."
        ) from exc


async def run_task(task_name: str, client: OpenAI) -> EpisodeOutcome:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[EcomEnv] = None

    log_start(task=task_name, env=BENCHMARK, model=_model_name())

    try:
        env_base_url = _env_base_url()
        image_name = _image_name()

        if env_base_url:
            env = EcomEnv(base_url=env_base_url)
            await env.connect()
        else:
            if not image_name:
                raise RuntimeError(
                    "IMAGE_NAME or LOCAL_IMAGE_NAME is required when ENV_BASE_URL is not set"
                )
            env = await EcomEnv.from_docker_image(image_name)

        result = await env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation
            action = get_model_action(client, step, observation, history)
            if action is None:
                action = heuristic_policy(observation, step)

            result = await env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = _extract_last_action_error(result.observation)

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=format_action(action),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {format_action(action)} -> reward {reward:.2f} error={error or 'null'}"
            )

            if done:
                info = result.observation.info
                if isinstance(info, dict):
                    success = bool(info.get("grader_success", False))
                    raw_score = info.get("grader_score", 0.0)
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = 0.0
                score = max(0.0, min(1.0, score))
                break

    except Exception:
        success = False
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return EpisodeOutcome(
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards,
    )


async def main() -> None:
    client = _build_llm_client()
    _probe_llm_proxy(client)

    if not _env_base_url() and not _image_name():
        raise RuntimeError(
            "Set ENV_BASE_URL or IMAGE_NAME/LOCAL_IMAGE_NAME before running inference.py"
        )

    for task_name in _task_names():
        await run_task(task_name, client)


if __name__ == "__main__":
    asyncio.run(main())
