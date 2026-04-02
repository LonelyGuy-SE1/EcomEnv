# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ecom returns decision environment client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EcomAction, EcomObservation


class EcomEnv(EnvClient[EcomAction, EcomObservation, State]):
    """Client for the returns decision environment."""

    def _step_payload(self, action: EcomAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }
        if action.reason_code is not None:
            payload["reason_code"] = action.reason_code
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[EcomObservation]:
        obs_data = payload.get("observation", {})
        observation = EcomObservation(
            return_reason=obs_data.get("return_reason", ""),
            product_category=obs_data.get("product_category", ""),
            product_value=obs_data.get("product_value", "low"),
            days_since_purchase=int(obs_data.get("days_since_purchase", 0)),
            user_account_age_days=int(obs_data.get("user_account_age_days", 0)),
            product_condition_notes=obs_data.get("product_condition_notes", ""),
            return_rate=float(obs_data.get("return_rate", 0.0)),
            total_orders=int(obs_data.get("total_orders", 1)),
            policy_summary=obs_data.get("policy_summary", ""),
            info=obs_data.get("info", {}),
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
        )
