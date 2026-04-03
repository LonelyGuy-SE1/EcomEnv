# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Returns decision environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ecom.models import EcomAction, EcomObservation, EcomReward

Difficulty = Literal["easy", "medium", "hard"]
Intent = Literal["genuine", "abusive"]
FinalAction = Literal["APPROVE", "REJECT", "ESCALATE"]


@dataclass(frozen=True)
class DifficultyConfig:
    fraud_probability: float
    ambiguity_rate: float
    conflict_rate: float


@dataclass(frozen=True)
class TaskSpec:
    difficulty: Difficulty
    seed: int
    objective: str
    success_threshold: float


@dataclass(frozen=True)
class PolicyProfile:
    window_days: int
    non_returnable: tuple[str, ...]
    exception_text: str

    def summary(self) -> str:
        categories = ", ".join(self.non_returnable)
        return (
            f"Returns accepted within {self.window_days} days. "
            f"Non-returnable categories: {categories}. "
            f"Exception: {self.exception_text}."
        )


@dataclass
class HiddenCaseState:
    fraud_risk_score: float
    true_intent: Intent
    optimal_action: FinalAction
    cost_impact: Dict[str, float]
    category_policy_violated: bool
    time_policy_violated: bool
    exception_applies: bool
    is_ambiguous: bool


@dataclass
class VisibleCase:
    return_reason: str
    product_category: str
    product_value: Literal["low", "medium", "high"]
    days_since_purchase: int
    user_account_age_days: int
    product_condition_notes: str
    return_rate: float
    total_orders: int
    policy_summary: str


class EcomEnvironment(Environment[EcomAction, EcomObservation, State]):
    """Single-request return decision environment with partial observability."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _MAX_STEPS: int = 4

    _VALUE_INDEX: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}

    _DIFFICULTY_DEFAULTS: Dict[Difficulty, DifficultyConfig] = {
        "easy": DifficultyConfig(
            fraud_probability=0.10, ambiguity_rate=0.10, conflict_rate=0.05
        ),
        "medium": DifficultyConfig(
            fraud_probability=0.25, ambiguity_rate=0.30, conflict_rate=0.20
        ),
        "hard": DifficultyConfig(
            fraud_probability=0.40, ambiguity_rate=0.55, conflict_rate=0.45
        ),
    }

    _CATEGORY_POLICIES: Dict[str, PolicyProfile] = {
        "electronics": PolicyProfile(
            window_days=30,
            non_returnable=("final-sale", "personal-care"),
            exception_text="Defective electronics remain returnable even beyond standard restrictions",
        ),
        "fashion": PolicyProfile(
            window_days=45,
            non_returnable=("underwear", "swimwear"),
            exception_text="Quality defects override category restrictions",
        ),
        "home": PolicyProfile(
            window_days=60,
            non_returnable=("custom-furniture",),
            exception_text="Damage in transit is always eligible for return",
        ),
    }

    _CONDITION_NOTES: Dict[str, tuple[str, str]] = {
        "defective": (
            "Customer reports device fails to power on intermittently",
            "Diagnostic notes show repeat hardware faults and consistent malfunction",
        ),
        "wrong-item": (
            "Packaging label and item description appear mismatched",
            "Warehouse scan and photo check confirm SKU mismatch from fulfillment",
        ),
        "damaged-shipping": (
            "Outer box shows dents and seal damage from transit",
            "Carrier handoff log notes impact event with photo-confirmed product damage",
        ),
        "changed-mind": (
            "Customer no longer wants the item and packaging appears opened",
            "Follow-up confirms item used lightly with no defect evidence",
        ),
        "size-issue": (
            "Customer reports fit mismatch after trying item once",
            "Follow-up confirms sizing mismatch with item otherwise in resellable condition",
        ),
    }

    _TASKS: Dict[str, TaskSpec] = {
        "easy_policy_compliance": TaskSpec(
            difficulty="easy",
            seed=111,
            objective=(
                "Handle a straightforward, low-risk return and maximize policy-compliant value."
            ),
            success_threshold=0.75,
        ),
        "medium_balanced_judgment": TaskSpec(
            difficulty="medium",
            seed=222,
            objective=(
                "Balance policy, fraud risk, and cost trade-offs in an ambiguous return request."
            ),
            success_threshold=0.68,
        ),
        "hard_conflicting_signals": TaskSpec(
            difficulty="hard",
            seed=333,
            objective=(
                "Resolve conflicting high-value risk signals with policy-compliant evidence handling."
            ),
            success_threshold=0.74,
        ),
    }

    def __init__(
        self,
        mode: Difficulty = "medium",
        *,
        fraud_probability: Optional[float] = None,
        ambiguity_rate: Optional[float] = None,
        conflict_rate: Optional[float] = None,
        task_name: Optional[str] = None,
    ):
        self._task_name: Optional[str] = task_name
        self._task_spec: Optional[TaskSpec] = None

        if task_name is not None:
            if task_name not in self._TASKS:
                valid = ", ".join(sorted(self._TASKS))
                raise ValueError(
                    f"Unknown task_name '{task_name}'. Valid tasks: {valid}"
                )
            self._task_spec = self._TASKS[task_name]
            mode = self._task_spec.difficulty

        if mode not in self._DIFFICULTY_DEFAULTS:
            raise ValueError("mode must be one of: easy, medium, hard")

        self._mode: Difficulty = mode
        base_cfg = self._DIFFICULTY_DEFAULTS[mode]
        self._cfg = DifficultyConfig(
            fraud_probability=self._clamp01(
                base_cfg.fraud_probability
                if fraud_probability is None
                else fraud_probability
            ),
            ambiguity_rate=self._clamp01(
                base_cfg.ambiguity_rate if ambiguity_rate is None else ambiguity_rate
            ),
            conflict_rate=self._clamp01(
                base_cfg.conflict_rate if conflict_rate is None else conflict_rate
            ),
        )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._visible_case: Optional[VisibleCase] = None
        self._hidden_case: Optional[HiddenCaseState] = None
        self._requested_info = False
        self._done = False
        self._task_seed: Optional[int] = (
            self._task_spec.seed if self._task_spec else None
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> EcomObservation:
        del kwargs
        if task_name is not None:
            if task_name not in self._TASKS:
                valid = ", ".join(sorted(self._TASKS))
                raise ValueError(
                    f"Unknown task_name '{task_name}'. Valid tasks: {valid}"
                )
            self._task_name = task_name
            self._task_spec = self._TASKS[task_name]
            self._task_seed = self._task_spec.seed
            self._mode = self._task_spec.difficulty
            base_cfg = self._DIFFICULTY_DEFAULTS[self._mode]
            self._cfg = DifficultyConfig(
                fraud_probability=base_cfg.fraud_probability,
                ambiguity_rate=base_cfg.ambiguity_rate,
                conflict_rate=base_cfg.conflict_rate,
            )

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._requested_info = False
        self._done = False

        effective_seed = seed
        if effective_seed is None and self._task_seed is not None:
            effective_seed = self._task_seed

        rng = self._rng(effective_seed)
        self._visible_case, self._hidden_case = self._generate_case(rng)

        return self._to_observation(
            self._visible_case,
            reward=None,
            done=False,
            info={
                "mode": self._mode,
                "task_name": self._task_name,
                "task_objective": self._task_spec.objective
                if self._task_spec
                else None,
                "task_seed": effective_seed,
                "phase": "initial",
                "step_contract": "observation_reward_done_info",
            },
        )

    def step(
        self,
        action: EcomAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> EcomObservation:
        del timeout_s, kwargs
        if self._visible_case is None or self._hidden_case is None:
            # Allow stateless HTTP /step calls by lazily initializing an episode.
            self.reset()
        if self._done:
            raise RuntimeError(
                "Episode already terminated. Call reset() to start a new episode"
            )

        if self._state.step_count >= self._MAX_STEPS:
            self._done = True
            timeout_info = {
                "mode": self._mode,
                "task_name": self._task_name,
                "phase": "terminal",
                "breakdown": EcomReward(
                    policy_gate=0.0,
                    financial_score=0.0,
                    fraud_score=0.0,
                    efficiency_score=0.0,
                    normalized_reward=0.0,
                    policy_violation=True,
                ).model_dump(),
                "grader_score": 0.0,
                "grader_success": False,
                "termination_reason": "max_steps_exceeded",
                "step_contract": "observation_reward_done_info",
            }
            return self._to_observation(
                self._visible_case,
                reward=0.0,
                done=True,
                info=timeout_info,
            )

        self._state.step_count += 1

        if action.action_type == "REQUEST_INFO":
            if self._requested_info:
                info = {
                    "invalid_action": "REQUEST_INFO already used",
                    "allowed_actions": ["APPROVE", "REJECT", "ESCALATE"],
                    "step_penalty": -0.10,
                    "step_contract": "observation_reward_done_info",
                }
                return self._to_observation(
                    self._visible_case,
                    reward=-0.10,
                    done=False,
                    info=info,
                )

            self._requested_info = True
            self._visible_case = self._refine_after_request_info(
                self._visible_case,
                self._hidden_case,
            )
            info_gain_reward = 0.08 if self._hidden_case.is_ambiguous else -0.03
            info = {
                "phase": "post_request_info",
                "revealed": ["product_condition_notes", "return_reason"],
                "step_reward": info_gain_reward,
                "step_contract": "observation_reward_done_info",
            }
            return self._to_observation(
                self._visible_case,
                reward=info_gain_reward,
                done=False,
                info=info,
            )

        if action.action_type not in ("APPROVE", "REJECT", "ESCALATE"):
            info = {
                "invalid_action": "Final action must be APPROVE, REJECT, or ESCALATE",
                "step_penalty": -0.05,
                "step_contract": "observation_reward_done_info",
            }
            return self._to_observation(
                self._visible_case,
                reward=-0.05,
                done=False,
                info=info,
            )

        reward, breakdown = self._evaluate(
            action, self._visible_case, self._hidden_case
        )
        self._done = True
        info = {
            "mode": self._mode,
            "task_name": self._task_name,
            "phase": "terminal",
            "breakdown": breakdown,
            "grader_score": float(breakdown["normalized_reward"]),
            "grader_success": self._task_success(float(breakdown["normalized_reward"])),
            "step_contract": "observation_reward_done_info",
        }
        return self._to_observation(
            self._visible_case,
            reward=reward,
            done=True,
            info=info,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            mode=self._mode,
            task_name=self._task_name,
            task_objective=self._task_spec.objective if self._task_spec else None,
            done=self._done,
            requested_info=self._requested_info,
        )

    @classmethod
    def task_names(cls) -> tuple[str, ...]:
        return tuple(cls._TASKS.keys())

    @classmethod
    def task_spec(cls, task_name: str) -> TaskSpec:
        if task_name not in cls._TASKS:
            valid = ", ".join(sorted(cls._TASKS))
            raise ValueError(f"Unknown task_name '{task_name}'. Valid tasks: {valid}")
        return cls._TASKS[task_name]

    def _task_success(self, score: float) -> bool:
        if self._task_spec is None:
            return score >= 0.7
        return score >= self._task_spec.success_threshold

    def grader_score(self, action: EcomAction) -> float:
        if self._visible_case is None or self._hidden_case is None:
            raise RuntimeError("Environment must be reset() before grader scoring")
        score, _ = self._evaluate(action, self._visible_case, self._hidden_case)
        return score

    def get_metadata(self) -> "EnvironmentMetadata":
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="ecom-returns-decision",
            description=(
                "Operational return-decision environment with policy constraints, "
                "latent fraud risk, and cost-aware grading."
            ),
            version="1.0.0",
            author="OpenEnv_H",
            documentation_url="https://huggingface.co/spaces/Lonelyguyse1/ecom",
        )

    @staticmethod
    def _rng(seed: Optional[int]):
        import random

        return random.Random(seed)

    def _generate_case(self, rng) -> tuple[VisibleCase, HiddenCaseState]:
        if self._task_name == "hard_conflicting_signals":
            return self._generate_hard_case(rng)

        category = rng.choice(tuple(self._CATEGORY_POLICIES.keys()))
        policy = self._CATEGORY_POLICIES[category]

        return_reason = self._weighted_choice(
            rng,
            {
                "defective": 0.24,
                "wrong-item": 0.14,
                "damaged-shipping": 0.12,
                "changed-mind": 0.28,
                "size-issue": 0.22,
            },
        )
        value_bucket = self._weighted_choice(
            rng,
            {
                "low": 0.40,
                "medium": 0.40,
                "high": 0.20,
            },
        )

        days_since_purchase = rng.randint(0, 90)
        user_account_age_days = rng.randint(15, 2200)
        total_orders = rng.randint(2, 220)

        # Mandatory behavioral signal.
        return_rate = self._sample_return_rate(rng, total_orders)

        # Mandatory correlations.
        # 1) Higher return_rate -> higher fraud, lower return_rate -> lower fraud.
        # 2) Higher product_value -> higher fraud, lower product_value -> lower fraud.
        base_risk = self._cfg.fraud_probability
        risk = base_risk
        risk += 0.35 * (return_rate - 0.30)
        risk += 0.10 * (self._VALUE_INDEX[value_bucket] - 1)
        risk += 0.08 if return_reason == "changed-mind" else 0.0
        risk -= (
            0.10
            if return_reason in ("defective", "wrong-item", "damaged-shipping")
            else 0.0
        )
        risk += 0.10 if user_account_age_days < 90 else 0.0
        fraud_risk_score = self._clamp01(risk)

        # Intent depends on computed latent risk, not independent coin flips.
        true_intent: Intent = (
            "abusive" if rng.random() < fraud_risk_score else "genuine"
        )

        # Policy constraints with exception support.
        exception_applies = self._exception_applies(category, return_reason)

        category_policy_violated = False
        if not exception_applies:
            category_flag_prob = 0.10 + 0.12 * self._cfg.conflict_rate
            if return_reason == "changed-mind":
                category_flag_prob += 0.10
            if rng.random() < self._clamp01(category_flag_prob):
                policy_tag = rng.choice(policy.non_returnable)
                category_policy_violated = True
            else:
                policy_tag = None
        else:
            policy_tag = None

        time_policy_violated = (
            days_since_purchase > policy.window_days and not exception_applies
        )

        is_ambiguous = rng.random() < self._cfg.ambiguity_rate
        is_conflicting = rng.random() < self._cfg.conflict_rate

        # Hardness is not just fraud probability; ambiguity and conflicts reshape signals.
        condition_brief, condition_detailed = self._CONDITION_NOTES[return_reason]
        if category_policy_violated and policy_tag is not None:
            condition_brief += (
                f"; order line is tagged under restricted class '{policy_tag}'"
            )
            condition_detailed += (
                f"; policy audit confirms '{policy_tag}' item class on this order line"
            )

        if is_conflicting:
            condition_brief = self._inject_conflict_signal(
                condition_brief, return_reason
            )
            condition_detailed = self._inject_conflict_signal(
                condition_detailed, return_reason
            )

        policy_summary = policy.summary()
        if is_ambiguous:
            policy_summary = self._make_policy_more_ambiguous(policy_summary)

        visible = VisibleCase(
            return_reason=return_reason,
            product_category=category,
            product_value=value_bucket,
            days_since_purchase=days_since_purchase,
            user_account_age_days=user_account_age_days,
            product_condition_notes=condition_brief,
            return_rate=return_rate,
            total_orders=total_orders,
            policy_summary=policy_summary,
        )

        financial_scores = self._financial_scores(
            value_bucket=value_bucket,
            intent=true_intent,
            category_violation=category_policy_violated,
            time_violation=time_policy_violated,
            return_reason=return_reason,
        )
        optimal_action = self._argmax_action(financial_scores)

        hidden = HiddenCaseState(
            fraud_risk_score=fraud_risk_score,
            true_intent=true_intent,
            optimal_action=optimal_action,
            cost_impact=financial_scores,
            category_policy_violated=category_policy_violated,
            time_policy_violated=time_policy_violated,
            exception_applies=exception_applies,
            is_ambiguous=is_ambiguous or is_conflicting,
        )
        return visible, hidden

    def _generate_hard_case(self, rng) -> tuple[VisibleCase, HiddenCaseState]:
        category = "electronics"
        policy = self._CATEGORY_POLICIES[category]

        return_reason = rng.choice(("wrong-item", "changed-mind"))
        value_bucket = "high"
        days_since_purchase = rng.randint(
            max(1, policy.window_days - 4), policy.window_days
        )
        user_account_age_days = rng.randint(25, 120)
        total_orders = rng.randint(4, 28)
        return_rate = self._clamp01(0.50 + rng.random() * 0.06)

        base_risk = self._cfg.fraud_probability
        risk = base_risk
        risk += 0.35 * (return_rate - 0.30)
        risk += 0.10 * (self._VALUE_INDEX[value_bucket] - 1)
        risk += 0.12 if return_reason in ("changed-mind", "wrong-item") else 0.0
        risk += 0.10 if user_account_age_days < 90 else 0.0
        fraud_risk_score = self._clamp01(risk)
        true_intent: Intent = (
            "abusive" if rng.random() < max(0.75, fraud_risk_score) else "genuine"
        )

        exception_applies = self._exception_applies(category, return_reason)
        category_policy_violated = (not exception_applies) and (rng.random() < 0.12)
        time_policy_violated = (
            days_since_purchase > policy.window_days and not exception_applies
        )

        condition_brief, condition_detailed = self._CONDITION_NOTES[return_reason]
        condition_brief += "; escalation history indicates conflicting evidence"
        condition_detailed += (
            "; forensic audit surfaced contradictory claimant and logistics signals"
        )
        condition_brief += (
            "; prior return investigations on this account show disputed evidence"
        )
        condition_detailed += "; cross-team review indicates conflicting intent indicators and high potential abuse exposure"

        if category_policy_violated:
            condition_brief += (
                "; order line is tagged under restricted class 'final-sale'"
            )
            condition_detailed += (
                "; policy audit confirms 'final-sale' restricted class"
            )

        policy_summary = (
            "Returns accepted within 30 days. "
            "Non-returnable categories: final-sale, personal-care. "
            "Exception: Defective electronics remain returnable even beyond standard restrictions. "
            "For conflicting claims, request additional evidence before any final decision."
        )

        visible = VisibleCase(
            return_reason=return_reason,
            product_category=category,
            product_value=value_bucket,
            days_since_purchase=days_since_purchase,
            user_account_age_days=user_account_age_days,
            product_condition_notes=condition_brief,
            return_rate=return_rate,
            total_orders=total_orders,
            policy_summary=policy_summary,
        )

        financial_scores = self._financial_scores(
            value_bucket=value_bucket,
            intent=true_intent,
            category_violation=category_policy_violated,
            time_violation=time_policy_violated,
            return_reason=return_reason,
        )

        if time_policy_violated or category_policy_violated:
            optimal_action: FinalAction = "REJECT"
        elif true_intent == "abusive" and fraud_risk_score >= 0.50:
            optimal_action = "REJECT"
        else:
            optimal_action = self._argmax_action(financial_scores)

        hidden = HiddenCaseState(
            fraud_risk_score=fraud_risk_score,
            true_intent=true_intent,
            optimal_action=optimal_action,
            cost_impact=financial_scores,
            category_policy_violated=category_policy_violated,
            time_policy_violated=time_policy_violated,
            exception_applies=exception_applies,
            is_ambiguous=True,
        )
        return visible, hidden

    @staticmethod
    def _sample_return_rate(rng, total_orders: int) -> float:
        band = rng.random()
        if band < 0.60:
            center = 0.12
            spread = 0.08
        elif band < 0.90:
            center = 0.30
            spread = 0.10
        else:
            center = 0.55
            spread = 0.12
        noise = (rng.random() * 2.0 - 1.0) * spread
        historical_pressure = min(0.08, 8.0 / float(total_orders))
        return EcomEnvironment._clamp01(center + noise + historical_pressure)

    @staticmethod
    def _inject_conflict_signal(text: str, reason: str) -> str:
        if reason in ("defective", "damaged-shipping"):
            return text + "; inspection has mixed indicators and partial evidence"
        return text + "; customer claims conflict with available logistics notes"

    @staticmethod
    def _make_policy_more_ambiguous(text: str) -> str:
        return (
            text + " In borderline cases, consistency checks and risk controls apply."
        )

    @staticmethod
    def _exception_reason_tokens(category: str) -> tuple[str, ...]:
        if category == "electronics":
            return ("defective",)
        if category == "fashion":
            return ("defective",)
        if category == "home":
            return ("damaged-shipping",)
        return ()

    def _exception_applies(self, category: str, return_reason: str) -> bool:
        return return_reason in self._exception_reason_tokens(category)

    @staticmethod
    def _financial_scores(
        *,
        value_bucket: str,
        intent: Intent,
        category_violation: bool,
        time_violation: bool,
        return_reason: str,
    ) -> Dict[FinalAction, float]:
        value_scale = {"low": 1.0, "medium": 1.7, "high": 2.6}[value_bucket]

        approve_gain = 0.45
        if return_reason in ("defective", "wrong-item", "damaged-shipping"):
            approve_gain += 0.25
        if intent == "abusive":
            approve_gain -= 0.45 * value_scale
        else:
            approve_gain -= 0.15 * value_scale
        if category_violation or time_violation:
            approve_gain -= 0.35

        reject_gain = 0.25
        if intent == "abusive":
            reject_gain += 0.35 * value_scale
        else:
            reject_gain -= 0.30
        if return_reason in ("defective", "wrong-item", "damaged-shipping"):
            reject_gain -= 0.20

        escalate_gain = 0.30
        escalate_gain -= 0.08 * value_scale
        if intent == "abusive":
            escalate_gain += 0.12
        if return_reason in ("defective", "damaged-shipping"):
            escalate_gain += 0.05

        return {
            "APPROVE": approve_gain,
            "REJECT": reject_gain,
            "ESCALATE": escalate_gain,
        }

    @staticmethod
    def _argmax_action(scores: Dict[FinalAction, float]) -> FinalAction:
        return max(scores.keys(), key=lambda k: scores[k])

    def _evaluate(
        self,
        action: EcomAction,
        visible: VisibleCase,
        hidden: HiddenCaseState,
    ) -> tuple[float, Dict[str, Any]]:
        policy_ok = self._policy_gate(action, visible, hidden)
        if not policy_ok:
            reward_model = EcomReward(
                policy_gate=0.0,
                financial_score=0.0,
                fraud_score=0.0,
                efficiency_score=0.0,
                normalized_reward=0.0,
                policy_violation=True,
            )
            return 0.0, reward_model.model_dump()

        final_action = action.action_type
        if final_action == "REJECT":
            reason_bonus = 0.0
            if hidden.time_policy_violated and action.reason_code == "TIME_EXPIRED":
                reason_bonus = 0.05
            elif (
                hidden.category_policy_violated
                and action.reason_code == "POLICY_VIOLATION"
            ):
                reason_bonus = 0.05
            elif (
                hidden.true_intent == "abusive"
                and action.reason_code == "SUSPECTED_FRAUD"
            ):
                reason_bonus = 0.05
        else:
            reason_bonus = 0.0

        trajectory_bonus = 0.0
        if self._requested_info and hidden.is_ambiguous:
            trajectory_bonus += 0.05
        elif hidden.is_ambiguous and not self._requested_info:
            trajectory_bonus -= 0.10

        # Component scores are individually bounded [0, 1] before weighting.
        financial_raw = (
            hidden.cost_impact[final_action] + reason_bonus + trajectory_bonus
        )
        financial_score = self._normalize_financial(financial_raw)

        fraud_score = self._fraud_component(final_action, hidden)
        efficiency_score = self._efficiency_component(final_action)

        final_reward = (
            0.50 * financial_score + 0.30 * fraud_score + 0.20 * efficiency_score
        )
        final_reward = self._clamp01(final_reward)

        reward_model = EcomReward(
            policy_gate=1.0,
            financial_score=financial_score,
            fraud_score=fraud_score,
            efficiency_score=efficiency_score,
            normalized_reward=final_reward,
            policy_violation=False,
            optimal_action=hidden.optimal_action,
            matched_optimal=final_action == hidden.optimal_action,
        )
        return final_reward, reward_model.model_dump()

    def _policy_gate(
        self,
        action: EcomAction,
        visible: VisibleCase,
        hidden: HiddenCaseState,
    ) -> bool:
        if (
            self._task_name == "hard_conflicting_signals"
            and hidden.is_ambiguous
            and not self._requested_info
            and action.action_type in ("APPROVE", "REJECT", "ESCALATE")
        ):
            return False

        if action.action_type == "APPROVE":
            if hidden.time_policy_violated or hidden.category_policy_violated:
                return False

            if (
                hidden.is_ambiguous
                and hidden.fraud_risk_score >= 0.55
                and not hidden.exception_applies
            ):
                return False

        if action.action_type == "REJECT":
            if hidden.time_policy_violated and hidden.category_policy_violated:
                if action.reason_code not in ("TIME_EXPIRED", "POLICY_VIOLATION"):
                    return False
            elif hidden.time_policy_violated:
                if action.reason_code != "TIME_EXPIRED":
                    return False
            elif hidden.category_policy_violated:
                if action.reason_code != "POLICY_VIOLATION":
                    return False
            else:
                if action.reason_code in ("TIME_EXPIRED", "POLICY_VIOLATION"):
                    return False

            # Prevent unsupported fraud accusation when fraud signal is very low.
            if (
                action.reason_code == "SUSPECTED_FRAUD"
                and hidden.fraud_risk_score < 0.45
            ):
                return False

        # If no violation and no fraud signal, rejecting a clear service-failure claim is policy-inconsistent.
        if (
            action.action_type == "REJECT"
            and not hidden.time_policy_violated
            and not hidden.category_policy_violated
            and hidden.fraud_risk_score < 0.30
            and visible.return_reason in ("defective", "wrong-item", "damaged-shipping")
        ):
            return False

        if (
            action.action_type == "ESCALATE"
            and hidden.is_ambiguous
            and not self._requested_info
        ):
            return False

        return True

    @staticmethod
    def _normalize_financial(raw_value: float) -> float:
        # Bound from approximately [-1.5, 1.5] into [0, 1] deterministically.
        return EcomEnvironment._clamp01((raw_value + 1.5) / 3.0)

    @staticmethod
    def _fraud_component(final_action: FinalAction, hidden: HiddenCaseState) -> float:
        risk = hidden.fraud_risk_score
        if final_action == "REJECT":
            if hidden.true_intent == "abusive":
                return EcomEnvironment._clamp01(0.60 + 0.40 * risk)
            return EcomEnvironment._clamp01(0.20 + 0.30 * (1.0 - risk))

        if final_action == "APPROVE":
            if hidden.true_intent == "genuine":
                return EcomEnvironment._clamp01(0.65 + 0.35 * (1.0 - risk))
            return EcomEnvironment._clamp01(0.10 + 0.20 * (1.0 - risk))

        # ESCALATE
        if hidden.true_intent == "abusive":
            return EcomEnvironment._clamp01(0.50 + 0.30 * risk)
        return EcomEnvironment._clamp01(0.45 + 0.25 * (1.0 - risk))

    def _efficiency_component(self, final_action: FinalAction) -> float:
        # Escalation and prior info requests incur efficiency penalty.
        base = 1.0
        if self._requested_info:
            base -= 0.20
        if final_action == "ESCALATE":
            base -= 0.30
        return self._clamp01(base)

    def _refine_after_request_info(
        self,
        visible: VisibleCase,
        hidden: HiddenCaseState,
    ) -> VisibleCase:
        reason = visible.return_reason
        if hidden.true_intent == "abusive":
            refined_reason = (
                "changed-mind" if reason in ("defective", "wrong-item") else reason
            )
            refined_notes = (
                visible.product_condition_notes
                + "; follow-up review found no reproducible defect evidence"
            )
        else:
            refined_reason = reason
            refined_notes = (
                self._CONDITION_NOTES[reason][1]
                if reason in self._CONDITION_NOTES
                else visible.product_condition_notes
            )

        # Deterministic, existing-field-only refinement.
        refined_return_rate = self._clamp01(
            visible.return_rate - 0.03
            if hidden.true_intent == "genuine"
            else visible.return_rate + 0.03
        )

        return VisibleCase(
            return_reason=refined_reason,
            product_category=visible.product_category,
            product_value=visible.product_value,
            days_since_purchase=visible.days_since_purchase,
            user_account_age_days=visible.user_account_age_days,
            product_condition_notes=refined_notes,
            return_rate=refined_return_rate,
            total_orders=visible.total_orders,
            policy_summary=visible.policy_summary,
        )

    @staticmethod
    def _to_observation(
        case: VisibleCase,
        *,
        reward: Optional[float],
        done: bool,
        info: Dict[str, Any],
    ) -> EcomObservation:
        return EcomObservation(
            return_reason=case.return_reason,
            product_category=case.product_category,
            product_value=case.product_value,
            days_since_purchase=case.days_since_purchase,
            user_account_age_days=case.user_account_age_days,
            product_condition_notes=case.product_condition_notes,
            return_rate=case.return_rate,
            total_orders=case.total_orders,
            policy_summary=case.policy_summary,
            reward=reward,
            done=done,
            info=info,
        )

    @staticmethod
    def _weighted_choice(rng, distribution: Dict[str, float]) -> str:
        threshold = rng.random()
        cumulative = 0.0
        last = None
        for key, weight in distribution.items():
            cumulative += weight
            last = key
            if threshold <= cumulative:
                return key
        assert last is not None
        return last

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return float(value)
