#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
python_executable="/home/yojiao/.conda/envs/py314/bin/python"
exec "$python_executable" \
  "$repo_root/scripts/jj_tms/render_jj_tms_hcp_fixed_template_overviews.py" \
  "$@"
