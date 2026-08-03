# syntax=docker/dockerfile:1.7
#
# MVP layer-reuse only: jobshift:latest is a locally-built base image that
# already carries Python 3.12 + uv + Torch 2.6.0+cu124 + Sentence Transformers.
# FRUS does not depend on Jobshift code or data; see README "Base image".
FROM jobshift:latest

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The base image ships its own source tree at /app. Left in place it becomes
# part of this project's working directory: `ruff format --check .` linted 11
# Jobshift files and reported a dirty tree. Clear it — FRUS shares the base
# image's layers, not its code.
RUN find /app -mindepth 1 -delete

# Resolve deps from the lock file. The bind mount reuses the host-side uv cache
# populated by `uv sync` so rebuilds work without network.
COPY pyproject.toml uv.lock ./
RUN --mount=type=bind,source=.uv-cache,target=/uv-cache,rw \
    UV_CACHE_DIR=/uv-cache uv sync --locked --no-install-project --all-groups

# README.md is not documentation here: `[project] readme` makes hatchling
# require it to build the package at all.
COPY README.md ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY eval/ ./eval/
RUN --mount=type=bind,source=.uv-cache,target=/uv-cache,rw \
    UV_CACHE_DIR=/uv-cache uv sync --locked --all-groups

ENV PATH="/opt/venv/bin:${PATH}"

CMD ["frus", "--help"]
