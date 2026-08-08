# syntax=docker/dockerfile:1.7
#
# Self-contained: a slim Python base plus exactly the dependencies in uv.lock.
#
# This used to build on jobshift:latest for "layer reuse". It reused nothing.
# Both images put their virtualenv at /opt/venv, so `uv sync` here overwrote the
# base image's — and because Docker layers are additive, overwriting a path in a
# later layer does not reclaim the earlier one. The image therefore carried two
# complete CUDA torch stacks, one of them unreachable: 6.4GB of shadowed base
# venv (torch 2.6.0+cu124) underneath the 5.8GB this project actually uses
# (torch 2.13.0+cu130). 19.2GB doing the work of about 6GB.
#
# No CUDA base image either. The torch wheels bundle their own CUDA runtime —
# that is what the 2.7GB of nvidia/* packages inside the venv are — and the
# driver comes from the host through `--gpus all`. A CUDA base image would add a
# second copy of the toolkit for nothing.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Resolve deps from the lock file before copying source, so editing a module
# does not invalidate the layer that takes minutes to build. The bind mount
# reuses the host-side uv cache so rebuilds work without network.
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
