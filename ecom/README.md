---
title: E-commerce Returns Decision Environment
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - operations
  - decision-making
---

# E-commerce Returns Decision Environment

This environment is a partially observable, policy-constrained decision process
for a single e-commerce return case per episode.

## Environment root and loader contract

This repository uses `ecom/` as the OpenEnv environment root.

- `ecom/openenv.yaml` is the authoritative manifest.
- `app: server.app:app` resolves inside `ecom/`, so it maps to
  `ecom/server/app.py`.
- Validate from repository root with `openenv validate ecom`, or from
  `ecom/` with `openenv validate .`.

## Formal task definition

Episode is defined over hidden state `s_t` and observation `o_t`.

- Hidden state contains fraud intent, policy violations, latent risk, and
  optimal action target.
- Observation exposes only operational case fields and policy summary text.
- Agent takes one action from `A = {APPROVE, REJECT, ESCALATE, REQUEST_INFO}`.
- Terminal objective is maximizing normalized reward while satisfying policy gate
  constraints.

The environment is not a static classifier. It is a short-horizon sequential
decision loop with action-dependent transition and scoring.

## Schemas

### Action (`EcomAction`)

- `action_type`: `APPROVE | REJECT | ESCALATE | REQUEST_INFO`
- `reason_code` is required only when `action_type == REJECT`
- Allowed reject reasons:
  - `TIME_EXPIRED`
  - `POLICY_VIOLATION`
  - `SUSPECTED_FRAUD`

Validation is strict: non-REJECT actions cannot carry `reason_code`.

### Observation (`EcomObservation`)

- `return_reason`
- `product_category`
- `product_value` in `{low, medium, high}`
- `days_since_purchase`
- `user_account_age_days`
- `product_condition_notes`
- `return_rate` in `[0,1]`
- `total_orders >= 1`
- `policy_summary`
- `reward`, `done`, `info`

### Reward payload (`EcomReward`)

Terminal breakdown keys:

- `policy_gate`
- `financial_score`
- `fraud_score`
- `efficiency_score`
- `normalized_reward`
- `policy_violation`
- `optimal_action`
- `matched_optimal`

All numeric reward components are bounded to `[0,1]`.

## Episode protocol

### Reset

`reset(seed=None, episode_id=None, task_name=None)`:

- initializes state
- samples deterministic or stochastic case
- returns initial observation with:
  - `info.phase=initial`
  - `info.available_actions=[APPROVE, REJECT, ESCALATE, REQUEST_INFO]`
  - `info.reject_reason_codes=[TIME_EXPIRED, POLICY_VIOLATION, SUSPECTED_FRAUD]`
  - `info.task_name`, `info.task_seed`, `info.task_objective` when task-based

### Step

`step(action)` follows these guards and transitions:

1. If called before reset:
   - action is ignored
   - returns fresh initial observation
   - sets `invalid_action` and `last_action_error` to
     `step_called_before_reset_action_ignored`
2. If called after terminal:
   - returns terminal observation
   - `reward=0.0`, `done=true`
   - sets `invalid_action` and `last_action_error` to
     `episode_already_terminated_call_reset`
3. `REQUEST_INFO` first use:
   - non-terminal
   - refines existing fields only
   - reward shaping: `+0.08` if ambiguous else `-0.03`
4. Repeated `REQUEST_INFO`:
   - non-terminal penalty `-0.10`
   - error code: `request_info_already_used`
5. Invalid non-terminal-final action type:
   - non-terminal penalty `-0.05`
   - error code: `invalid_final_action`
6. Valid terminal action (`APPROVE|REJECT|ESCALATE`):
   - runs policy gate then reward model
   - returns terminal observation with grader fields

Hard cap is `_MAX_STEPS=4`. Exceeding cap returns terminal `0.0` with
`termination_reason=max_steps_exceeded`.

## Info-channel contract

`info` is the machine-readable control channel. It is used for policy hints,
error handling, and grader reporting.

Common keys by phase:

- Initial phase:
  - `phase=initial`
  - `available_actions`
  - `reject_reason_codes`
- Post-`REQUEST_INFO` phase:
  - `phase=post_request_info`
  - `revealed`
  - `available_actions`
  - `reject_reason_codes`
- Terminal phase:
  - `phase=terminal`
  - `breakdown`
  - `grader_score`
  - `grader_success`
- Invalid action paths:
  - `invalid_action` (stable machine code)
  - `last_action_error` (same machine code)

## Case generation model

### Difficulty presets

- `easy`: fraud `0.10`, ambiguity `0.10`, conflict `0.05`
- `medium`: fraud `0.25`, ambiguity `0.30`, conflict `0.20`
- `hard`: fraud `0.40`, ambiguity `0.55`, conflict `0.45`

### Latent risk construction

For non-hard-template episodes, latent fraud risk is derived from correlated
signals, not independent labels.

Base formula (clamped to `[0,1]`):

```text
risk = base_fraud_probability
     + 0.35 * (return_rate - 0.30)
     + 0.10 * value_index
     + reason_and_account_adjustments
```

Where `value_index` maps low/medium/high to `-1/0/+1` offset through internal
indexing. Intent is then sampled from this latent risk.

### Policy model

Each category defines:

- return window days
- non-returnable category list
- exception text

Policy violations are split into:

- `time_policy_violated`
- `category_policy_violated`

Exception handling is explicitly modeled and influences both generation and
policy gate decisions.

### Ambiguity and conflict injection

- Ambiguity and conflict are sampled from difficulty-controlled rates.
- Conflict mutates condition/policy wording to create realistic contradictory
  evidence patterns.

### Hard template (`hard_conflicting_signals`)

The hard task uses a deterministic high-risk template:

- high-value electronics focus
- near-window timing
- intentionally conflicting evidence phrases
- stricter policy-gate behavior requiring evidence handling before finalization

## Transition semantics for `REQUEST_INFO`

`REQUEST_INFO` does not add new fields. It only refines existing observable
fields deterministically from hidden intent:

- `product_condition_notes`
- `return_reason` (may refine)
- `return_rate` (small deterministic shift)

This keeps schema fixed while allowing information-gathering behavior.

## Policy gate

If policy gate fails, terminal reward is forced to `0.0`.

Core constraints enforced:

- `APPROVE` is blocked on time/category violations.
- `APPROVE` may be blocked in high-risk ambiguous cases without exception.
- `REJECT` requires reason-code consistency with actual violation structure.
- Fraud rejection is blocked when fraud signal is too low.
- Rejecting clear low-fraud service-failure claims is blocked.
- In ambiguous hard scenarios, direct finalization before evidence collection can
  be blocked.

## Reward model

After gate pass:

1. Financial component:

```text
financial_raw = cost_impact[action] + reason_bonus + trajectory_bonus
financial_score = clamp01((financial_raw + 1.5) / 3.0)
```

2. Fraud component uses action-intent-risk-conditioned piecewise scoring.

3. Efficiency component:

```text
efficiency = 1.0 - 0.20*(requested_info_used) - 0.30*(action==ESCALATE)
```

4. Final reward:

```text
reward = clamp01(
    0.50 * financial_score
  + 0.30 * fraud_score
  + 0.20 * efficiency_score
)
```

Trajectory shaping:

- positive bonus for requesting info in ambiguous cases
- penalty for skipping info in ambiguous cases

## Deterministic task set

Tasks are fixed-name benchmarks with fixed seed and threshold:

1. `easy_policy_compliance`:
   - seed `111`
   - threshold `0.75`
2. `medium_balanced_judgment`:
   - seed `222`
   - threshold `0.68`
3. `hard_conflicting_signals`:
   - seed `333`
   - threshold `0.74`

Terminal `grader_success` is computed against the active task threshold.

## Determinism and reproducibility

- Uses `random.Random(seed)` for case generation.
- Task mode pins seed unless an explicit seed override is passed.
- No wall-clock dependence in generation or scoring.
- `grader_score(action)` is deterministic for a fixed latent case.

## Inference contract (`../inference.py`)

Baseline runner enforces strict one-line logs:

- `[START] task=<task> env=<benchmark> model=<model>`
- `[STEP] step=<n> action=<action> reward=<r> done=<bool> error=<value|null>`
- `[END] success=<bool> steps=<n> rewards=<r1,r2,...>`

Action selection path uses environment-provided control hints:

- `available_actions`
- `reject_reason_codes`
- `invalid_action` / `last_action_error`

This reduces invalid-action loops and keeps inference behavior aligned with
runtime contract.

## Validation checklist

From repository root:

```bash
openenv validate ecom
python -m pytest tests -q
./validate-submission.sh <space-url> .
```

From `ecom/`:

```bash
openenv validate .
openenv push
```
