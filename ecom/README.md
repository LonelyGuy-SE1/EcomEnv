---
title: E-commerce Returns Decision Environment
emoji: 📦
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

This environment simulates a real operations workflow in online retail: deciding how to handle customer return requests under policy constraints, latent fraud risk, and financial trade-offs.

It is designed as a **partially observable decision problem**, not a classification toy.

## Why this environment matters

Returns handling is a major cost center in e-commerce.

In production settings, an operations associate (or AI agent) must balance:

- customer satisfaction,
- policy compliance,
- fraud prevention,
- and cost efficiency.

This environment captures that exact tension with structured observations, hidden variables, and deterministic graders.

## Environment API (OpenEnv)

The environment follows the OpenEnv simulation API:

- `reset(...)` -> initial observation
- `step(action)` -> observation, reward, done, info
- `state` -> current episode state

The `step` info channel is exposed via `observation.info`.

## Action space

`EcomAction`:

- `action_type`: one of `APPROVE`, `REJECT`, `ESCALATE`, `REQUEST_INFO`
- `reason_code`: required only when `action_type == REJECT`
  - `TIME_EXPIRED`
  - `POLICY_VIOLATION`
  - `SUSPECTED_FRAUD`

## Observation space

`EcomObservation` fields:

- `return_reason`
- `product_category`
- `product_value` (`low | medium | high`)
- `days_since_purchase`
- `user_account_age_days`
- `product_condition_notes`
- `return_rate` (0.0 to 1.0)
- `total_orders`
- `policy_summary` (plain text, includes rules and exceptions)
- `info` (step metadata)

No identifier-only fields are included in the observation.

## Hidden state (grader-only)

The environment keeps the following latent variables hidden from the agent:

- `fraud_risk_score`
- `true_intent` (`genuine` or `abusive`)
- `cost_impact` by candidate action
- `optimal_action`

These are used to compute scores/rewards and evaluate decision quality.

## Episode flow and boundaries

- One request per episode.
- `APPROVE`, `REJECT`, `ESCALATE` are terminal actions (`done=True`).
- `REQUEST_INFO` is non-terminal on first use and deterministically refines existing observation fields:
  - `product_condition_notes`
  - `return_reason` (optional refinement)
  - slight refinement of `return_rate`
- No new fields are introduced after `REQUEST_INFO`.

## Scenario generation

Scenarios are generated programmatically from controlled distributions.

The generator includes mandatory realism correlations:

- higher `return_rate` -> higher fraud likelihood,
- lower `return_rate` -> lower fraud likelihood,
- higher `product_value` -> higher fraud likelihood,
- lower `product_value` -> lower fraud likelihood.

Difficulty is not just fraud probability; it also changes ambiguity and signal conflict.

## Reward design

Reward is deterministic and normalized to `[0.0, 1.0]`.

1. **Policy gate** (hard constraint)
   - policy violation => reward `0.0`
2. Component scores are bounded independently:
   - `financial_score in [0,1]`
   - `fraud_score in [0,1]`
   - `efficiency_score in [0,1]`
3. Weighted final score:
   - `0.5 * financial + 0.3 * fraud + 0.2 * efficiency`

This avoids overflow and grader instability.

## Tasks and graders (easy -> medium -> hard)

The environment ships with 3 deterministic benchmark tasks, each with fixed seed + threshold:

1. `easy_policy_compliance`
   - clear low-risk case
   - success threshold: `0.75`
2. `medium_balanced_judgment`
   - ambiguous policy/risk trade-off
   - success threshold: `0.68`
3. `hard_conflicting_signals`
   - high-value conflicting signals + exception pressure
   - success threshold: `0.62`

Terminal observation includes grader outputs in `info`:

- `grader_score` (0.0 to 1.0)
- `grader_success` (bool)
- detailed component `breakdown`

## Quick start

### Local dev server

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Python usage

```python
import asyncio
from ecom import EcomAction, EcomEnv


async def run():
    env = await EcomEnv.from_docker_image("ecom-env:latest")
    try:
        result = await env.reset(task_name="medium_balanced_judgment")
        # optional extra context
        result = await env.step(EcomAction(action_type="REQUEST_INFO"))
        # final decision
        result = await env.step(EcomAction(action_type="REJECT", reason_code="SUSPECTED_FRAUD"))
        print(result.reward, result.done, result.observation.info)
    finally:
        await env.close()


asyncio.run(run())
```

## Baseline inference

`inference.py` is at repo root as required.

Required env vars:

- `MODEL_NAME`
- `LOCAL_IMAGE_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`)

Optional:

- `API_BASE_URL` (defaults to `https://api.openai.com/v1`)

Run:

```bash
python inference.py
```

The script emits strict structured logs:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

### Reproducible baseline scores

Current deterministic baseline (heuristic fallback) on default task seeds:

- `easy_policy_compliance`: `0.7997`
- `medium_balanced_judgment`: `0.8388`
- `hard_conflicting_signals`: `0.8253`

## Hugging Face Spaces deployment

From `ecom/`:

```bash
openenv push
```

Or explicit options:

```bash
openenv push --repo-id <namespace>/<space-name> --private
```

## Docker

Build from the environment root (`ecom/`):

```bash
docker build -t ecom-env:latest -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 ecom-env:latest
```

Health check:

```bash
curl http://localhost:8000/health
```

## Validation

From `ecom/`:

```bash
openenv validate .
```

Optional pre-check from repository root:

```bash
./validate-submission.sh <your-space-url> .
```
