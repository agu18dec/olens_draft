#!/usr/bin/env bash
# One-command environment bootstrap for the oracle-lens repo.
#
# Creates/updates the repo-local .venv from pyproject.toml + uv.lock. Deliberately pins
# UV_PROJECT_ENVIRONMENT to this repo's .venv so it can never reconcile a venv elsewhere
# (e.g. a shared cluster venv pointed at by an inherited UV_PROJECT_ENVIRONMENT).
#
# Usage: bash setup.sh [--dev]   # --dev also installs ruff/mypy/pytest/matplotlib

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f pyproject.toml ] || ! grep -q '^name = "oracle-lens"' pyproject.toml; then
  echo "setup.sh must run from the oracle-lens repo root" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
unset UV_NO_SYNC 2>/dev/null || true

if [ "${1:-}" = "--dev" ]; then
  uv sync --group dev
else
  uv sync
fi
echo "OK — venv at $UV_PROJECT_ENVIRONMENT. Run things with: uv run <cmd>"
