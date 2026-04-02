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

from openenv.core.client_types import StepResult

from ecom import EcomAction, EcomEnv
from ecom.server.ecom_environment import EcomEnvironment

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


class LocalEnvRunner:
    """Fallback runner when Docker or remote endpoint is unavailable."""

    def __init__(self, mode: str = "medium"):
        self._env = EcomEnvironment(mode=mode)  # type: ignore[arg-type]

    async def reset(self, **kwargs: Any) -> StepResult[Any]:
        obs = self._env.reset(**kwargs)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def step(self, action: EcomAction, **kwargs: Any) -> StepResult[Any]:
        del kwargs
        obs = self._env.step(action)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def close(self) -> None:
        self._env.close()


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
    value = info.get("last_action_error")
    if value is None:
        return None
    return str(value)


def heuristic_policy(observation: Any, step: int) -> EcomAction:
    window = extract_return_window(observation.policy_summary)
    notes = observation.product_condition_notes.lower()
    reason = observation.return_reason
    return_rate = float(observation.return_rate)

    if observation.days_since_purchase > window:
        return EcomAction(action_type="REJECT", reason_code="TIME_EXPIRED")

    if "restricted class" in notes:
        return EcomAction(action_type="REJECT", reason_code="POLICY_VIOLATION")

    if return_rate >= 0.60 and observation.product_value == "high":
        return EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD")

    if reason in ("defective", "wrong-item", "damaged-shipping") and return_rate < 0.55:
        return EcomAction(action_type="APPROVE")

    ambiguous = (
        ("mixed indicators" in notes)
        or ("conflict" in notes)
        or (0.40 <= return_rate <= 0.60)
    )
    if step == 1 and ambiguous:
        return EcomAction(action_type="REQUEST_INFO")

    if return_rate >= 0.55:
        return EcomAction(action_type="ESCALATE")
    return EcomAction(action_type="APPROVE")


def model_policy(
    client: Optional[Any], observation: Any, step: int
) -> Optional[EcomAction]:
    if client is None:
        return None

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
            return EcomAction(action_type="REJECT", reason_code=reason_code)

        if action_type in {"APPROVE", "ESCALATE", "REQUEST_INFO"}:
            return EcomAction(action_type=action_type)
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
        # Fallback to deterministic local execution for reproducible baseline.
        env = LocalEnvRunner(mode="medium")
        result = await env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            observation = result.observation
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

    for task_name in TASKS:
        await run_task(task_name, client)


if __name__ == "__main__":
    asyncio.run(main())
