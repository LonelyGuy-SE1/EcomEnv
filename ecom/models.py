# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the returns decision environment."""

from typing import Any, Dict, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator

ActionType = Literal["APPROVE", "REJECT", "ESCALATE", "REQUEST_INFO"]
RejectReason = Literal["TIME_EXPIRED", "POLICY_VIOLATION", "SUSPECTED_FRAUD"]
ValueBucket = Literal["low", "medium", "high"]


class EcomAction(Action):
    """Action schema for return-request handling."""

    action_type: ActionType = Field(..., description="Decision type")
    reason_code: Optional[RejectReason] = Field(
        default=None,
        description="Required when action_type is REJECT",
    )

    @model_validator(mode="after")
    def validate_reason_code(self) -> "EcomAction":
        if self.action_type == "REJECT" and self.reason_code is None:
            raise ValueError("reason_code is required when action_type is REJECT")
        if self.action_type != "REJECT" and self.reason_code is not None:
            raise ValueError("reason_code is only allowed when action_type is REJECT")
        return self


class EcomObservation(Observation):
    """Observation schema for the partially observable returns task."""

    return_reason: str = Field(..., description="Customer-provided return reason")
    product_category: str = Field(..., description="Product category")
    product_value: ValueBucket = Field(..., description="Value bucket")
    days_since_purchase: int = Field(..., ge=0, description="Elapsed days")
    user_account_age_days: int = Field(..., ge=0, description="Account age in days")
    product_condition_notes: str = Field(..., description="Condition summary")
    return_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Historical return rate"
    )
    total_orders: int = Field(..., ge=1, description="Total historical orders")
    policy_summary: str = Field(
        ...,
        description="Natural-language policy text with rules and exceptions",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step info payload (OpenEnv-compatible info channel)",
    )


class EcomReward(BaseModel):
    """Typed reward breakdown used by deterministic task graders."""

    policy_gate: float = Field(..., ge=0.0, le=1.0)
    financial_score: float = Field(..., ge=0.0, le=1.0)
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    normalized_reward: float = Field(..., ge=0.0, le=1.0)
    policy_violation: bool
    optimal_action: Optional[str] = None
    matched_optimal: Optional[bool] = None
