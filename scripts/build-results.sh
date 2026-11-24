#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
project_python="${CS6220_PYTHON:-.venv/bin/python}"
"$project_python" scripts/write_result_tables.py --root .
mkdir -p results/report-build
if [[ -n "${CS6220_TECTONIC:-}" ]]; then
    project_tex_engine="$CS6220_TECTONIC"
elif command -v tectonic >/dev/null 2>&1; then
    project_tex_engine="$(command -v tectonic)"
else
    project_tex_engine="../../ccfa-workfiles/planning/cs6220-proposal/cache/tools/tectonic"
fi
cd docs
"$project_tex_engine" --keep-logs --outdir ../results/report-build initial-results.tex
cp -- ../results/report-build/initial-results.pdf initial-results.pdf
