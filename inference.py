"""Baseline inference runner for the Ecom returns decision environment.

This script follows the required structured stdout format:
  [START] ...
  [STEP]  ...
  [END]   ...
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ecom import EcomAction, EcomEnv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = OPENAI_API_KEY or HF_TOKEN or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")

BENCHMARK = os.getenv("ECOM_BENCHMARK", "ecom_returns_decision")
MAX_STEPS = 3
MAX_TOKENS = 180

TASKS: List[str] = [
    "easy_policy_compliance",
    "medium_balanced_judgment",
    "hard_conflicting_signals",
]


@dataclass
class EpisodeOutcome:
    success: bool
    steps: int
    rewards: List[float]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    done_val = str(done).lower()
    error_val = "null" if error is None else error
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True
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
    text = str(observation.product_condition_notes).lower()
    return "restricted class" in text


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
    if not hasattr(observation, "info"):
        return None
    info = observation.info
    if not isinstance(info, dict):
        return None
    for key in ("last_action_error", "invalid_action"):
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_available_actions(observation: Any) -> List[str]:
    if not hasattr(observation, "info"):
        return []
    info = observation.info
    if not isinstance(info, dict):
        return []
    raw = info.get("available_actions")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


def _extract_reject_reason_codes(observation: Any) -> List[str]:
    if not hasattr(observation, "info"):
        return []
    info = observation.info
    if not isinstance(info, dict):
        return []
    raw = info.get("reject_reason_codes")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


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
    notes = observation.product_condition_notes.lower()
    reason = observation.return_reason
    return_rate = float(observation.return_rate)

    ambiguous = (
        ("mixed indicators" in notes)
        or ("conflict" in notes)
        or (0.40 <= return_rate <= 0.65)
        or (observation.days_since_purchase > window and has_exception)
    )
    if step == 1 and ambiguous:
        if available_actions and "REQUEST_INFO" not in available_actions:
            pass
        else:
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


def model_policy(
    client: Optional[Any], observation: Any, step: int
) -> Optional[EcomAction]:
    if client is None:
        return None

    available_actions = _extract_available_actions(observation)
    reject_reason_codes = _extract_reject_reason_codes(observation)

    prompt = (
        "You are a returns operations agent. Choose one action JSON only.\n"
        "Allowed action_type: APPROVE, REJECT, ESCALATE, REQUEST_INFO\n"
        "If action_type is REJECT, include reason_code with one of: "
        "TIME_EXPIRED, POLICY_VIOLATION, SUSPECTED_FRAUD\n"
        "Output ONLY JSON, no prose.\n\n"
        f"Step: {step}\n"
        f"return_reason: {observation.return_reason}\n"
        f"product_category: {observation.product_category}\n"
        f"product_value: {observation.product_value}\n"
        f"days_since_purchase: {observation.days_since_purchase}\n"
        f"user_account_age_days: {observation.user_account_age_days}\n"
        f"product_condition_notes: {observation.product_condition_notes}\n"
        f"return_rate: {observation.return_rate:.3f}\n"
        f"total_orders: {observation.total_orders}\n"
        f"policy_summary: {observation.policy_summary}\n"
    )

    if available_actions:
        prompt += f"available_actions: {', '.join(available_actions)}\n"
    if reject_reason_codes:
        prompt += f"available_reject_reason_codes: {', '.join(reject_reason_codes)}\n"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()
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
    except Exception:
        return None

    return None


async def run_task(task_name: str, client: Optional[Any]) -> EpisodeOutcome:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    env: Optional[Any] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        if ENV_BASE_URL:
            env = EcomEnv(base_url=ENV_BASE_URL)
            await env.connect()
        else:
            if not LOCAL_IMAGE_NAME:
                raise RuntimeError(
                    "LOCAL_IMAGE_NAME is required when ENV_BASE_URL is not set"
                )
            env = await EcomEnv.from_docker_image(LOCAL_IMAGE_NAME)

        result = await env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            observation = result.observation

            action = model_policy(client, observation, step)
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

            if done:
                success = bool(result.observation.info.get("grader_success", False))
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

    return EpisodeOutcome(success=success, steps=steps_taken, rewards=rewards)


async def main() -> None:
    client = None
    if OpenAI is not None and API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    if not ENV_BASE_URL and not LOCAL_IMAGE_NAME:
        raise RuntimeError(
            "Set ENV_BASE_URL or LOCAL_IMAGE_NAME before running inference.py"
        )

    for task_name in TASKS:
        await run_task(task_name, client)


if __name__ == "__main__":
    asyncio.run(main())
