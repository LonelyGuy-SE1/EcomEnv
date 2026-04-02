# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the returns decision environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

import os


from ecom.models import EcomAction, EcomObservation
from ecom.server.ecom_environment import EcomEnvironment


def _env_factory() -> EcomEnvironment:
    mode = os.getenv("ECOM_MODE", "medium").strip().lower()
    if mode not in {"easy", "medium", "hard"}:
        mode = "medium"

    def _maybe_float(name: str) -> float | None:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return None
        return float(raw)

    return EcomEnvironment(
        mode=mode,
        fraud_probability=_maybe_float("ECOM_FRAUD_PROBABILITY"),
        ambiguity_rate=_maybe_float("ECOM_AMBIGUITY_RATE"),
        conflict_rate=_maybe_float("ECOM_CONFLICT_RATE"),
    )


# Create the app with web interface and README integration
app = create_app(
    _env_factory,
    EcomAction,
    EcomObservation,
    env_name="ecom",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m ecom.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn ecom.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
