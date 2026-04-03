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

## Problem definition

This environment models operational return-handling decisions for online retail.
An agent receives a partially observable request context and must choose one of:

- `APPROVE`
- `REJECT` with required reason code
- `ESCALATE`
- `REQUEST_INFO`

The objective is to optimize decision quality under policy constraints, fraud risk,
and cost trade-offs.

This is a decision environment, not a static classification benchmark.

## OpenEnv API contract

The environment implements the OpenEnv simulation interface:

- `reset(...) -> observation`
- `step(action) -> observation` where observation carries `reward`, `done`, and `info`
- `state -> State`

OpenEnv metadata is defined in `openenv.yaml`.

Validation:

```bash
openenv validate .
```

## Typed schemas

### Action (`EcomAction`)

- `action_type`: `APPROVE | REJECT | ESCALATE | REQUEST_INFO`
- `reason_code` (required only for `REJECT`):
  - `TIME_EXPIRED`
  - `POLICY_VIOLATION`
  - `SUSPECTED_FRAUD`

### Observation (`EcomObservation`)

- `return_reason`
- `product_category`
- `product_value` (`low | medium | high`)
- `days_since_purchase`
- `user_account_age_days`
- `product_condition_notes`
- `return_rate` (bounded `[0,1]`)
- `total_orders`
- `policy_summary` (text policy including exception clauses)
- `info` (step metadata and grader payload)

### Reward breakdown (`EcomReward`)

Terminal grader payload is typed and bounded:

- `policy_gate`
- `financial_score`
- `fraud_score`
- `efficiency_score`
- `normalized_reward`
- `policy_violation`
- `optimal_action`
- `matched_optimal`

Each numeric component is constrained to `[0,1]`.

## State and episode flow

### Episode semantics

- One return request per episode.
- Terminal actions: `APPROVE`, `REJECT`, `ESCALATE`.
- `REQUEST_INFO` is non-terminal on first use and refines existing fields only.
- Repeating `REQUEST_INFO` yields a penalty.
- Invalid final-action sequencing yields a penalty.
- Hard cap `_MAX_STEPS = 4`; exceeding cap terminates episode with zero score.

### `REQUEST_INFO` behavior

After `REQUEST_INFO`, the observation deterministically refines:

- `product_condition_notes`
- `return_reason` (may refine)
- `return_rate` (slight adjustment)

No new observation fields are introduced.

## Scenario generation

Scenarios are generated from controlled distributions.

### Global realism constraints

- higher `return_rate` increases latent fraud risk
- lower `return_rate` decreases latent fraud risk
- higher `product_value` increases latent fraud risk
- lower `product_value` decreases latent fraud risk

### Policy modeling

Category policies include:

- return window (days)
- non-returnable categories
- category-specific exceptions expressed in `policy_summary`

Exception application is aligned with category policy text.

## Task set and deterministic graders

The environment defines three deterministic benchmark tasks:

1. `easy_policy_compliance`
2. `medium_balanced_judgment`
3. `hard_conflicting_signals`

Each task has:

- fixed seed
- explicit objective string
- fixed success threshold

Hard mode uses conflict-heavy, high-value templates and enforces evidence-driven
handling in ambiguous cases.

## Reward function

Terminal reward is computed as:

1. **Policy gate**: violation => reward `0.0`
2. Component scoring (each independently bounded):
   - `financial_score in [0,1]`
   - `fraud_score in [0,1]`
   - `efficiency_score in [0,1]`
3. Weighted aggregate:

```text
reward = 0.5 * financial_score + 0.3 * fraud_score + 0.2 * efficiency_score
```

Additional shaping:

- positive/negative non-terminal reward on `REQUEST_INFO` depending on ambiguity
- penalties for invalid step patterns

## Inference script and reproducibility

Root `inference.py` is the submission baseline runner.

### Required environment variables

- `MODEL_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`)
- `LOCAL_IMAGE_NAME` when using `from_docker_image()`

Optional:

- `API_BASE_URL` (default: `https://api.openai.com/v1`)
- `ENV_BASE_URL` (remote environment endpoint)

### LLM client requirement

All LLM calls use:

```python
from openai import OpenAI
```

with `api_key` and `base_url` derived from environment variables.

### STDOUT log contract

The script emits strict one-line records:

- `[START] task=<task> env=<benchmark> model=<model>`
- `[STEP] step=<n> action=<action> reward=<r> done=<bool> error=<value|null>`
- `[END] success=<bool> steps=<n> rewards=<r1,r2,...>`

Rewards are formatted to two decimals.

### Current deterministic baseline (default seeds)

- `easy_policy_compliance`: `0.7997`
- `medium_balanced_judgment`: `0.8863` (`0.08 + 0.81` trajectory)
- `hard_conflicting_signals`: `0.9750` (`0.08 + 0.90` trajectory)

## Setup and execution

### Local server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker build and run

From `ecom/`:

```bash
docker build -t ecom-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 ecom-env:latest
```

Health check:

```bash
curl http://localhost:8000/health
```

### OpenEnv validation

From `ecom/`:

```bash
openenv validate .
```

From repository root (pre-check helper):

```bash
./validate-submission.sh <space-url> .
```

### Hugging Face deployment

From `ecom/`:

```bash
openenv push
```
