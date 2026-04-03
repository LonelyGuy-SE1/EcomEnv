# E-commerce Returns Decision Environment (OpenEnv)

This repository contains an OpenEnv-compatible environment for evaluating
cost-aware return-handling decisions under policy constraints and latent fraud risk.

Primary environment directory:

- `ecom/`

Key artifacts:

- Environment implementation: `ecom/server/ecom_environment.py`
- Typed models: `ecom/models.py`
- API app entrypoint: `ecom/server/app.py`
- OpenEnv manifest: `ecom/openenv.yaml`
- Detailed technical documentation: `ecom/README.md`
- Baseline inference runner: `inference.py`

Quick validation:

```bash
openenv validate ecom
```

Quick deployment:

```bash
cd ecom
openenv push
```
