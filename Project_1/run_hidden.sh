#!/usr/bin/env bash
# Hidden-test entry point: bash run_hidden.sh <input.jsonl> <output.jsonl>
# Needs LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in the environment (or in .env).
# Contracts (CUAD): hybrid pipeline, config FINAL (hybrid/pipeline.py), declared threshold 0.95.
# Forms/scans (ExtractBench): hybrid/forms.py config forms_final (two vision reads: LLM_MODEL + FORMS_SECOND_MODEL,
#   default gpt-5.5). HYBRID_NONCONTRACT=review sends every form field to a human instead.
# The original pipeline is still available: python -m solution.run --data IN --out OUT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY="${PYTHON:-python3}"; fi
exec "$PY" -m hybrid.run --data "$1" --out "$2" --config FINAL --trace "${2%.jsonl}.trace.jsonl"
