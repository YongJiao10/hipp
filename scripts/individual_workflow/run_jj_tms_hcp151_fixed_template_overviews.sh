#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
python_executable="/home/yojiao/.conda/envs/py314/bin/python"
exec "$python_executable" \
  "$repo_root/scripts/individual_workflow/render_jj_tms_hcp151_fixed_template_overviews.py" \
  "$@"
