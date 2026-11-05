#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
dataset="$(printf '%s' "${DATASET:-jj_tms}" | tr '[:upper:]-' '[:lower:]_')"
stanford_variant="${STANFORD_VARIANT:-}"
stanford_variant="$(printf '%s' "${stanford_variant}" | tr '[:upper:]-' '[:lower:]_')"
estt_variant="${ESTT_VARIANT:-}"
estt_variant="$(printf '%s' "${estt_variant}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_branch="${IOWA7T_BRANCH:-${BRANCH:-}}"
iowa7t_branch="$(printf '%s' "${iowa7t_branch}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_variant="${IOWA7T_VARIANT:-}"
iowa7t_variant="$(printf '%s' "${iowa7t_variant}" | tr '[:upper:]-' '[:lower:]_')"
case "${dataset}" in
  jj_tms)
    out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022}"
    ;;
  stanford)
    if [ -z "${stanford_variant}" ]; then
      stanford_variant="36p"
    fi
    case "${stanford_variant}" in
      36p)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022}"
        ;;
      *)
        echo "ERROR: STANFORD_VARIANT must be 36p, got: ${STANFORD_VARIANT:-}" >&2
        exit 2
        ;;
    esac
    ;;
  iowa7t)
    if [ -z "${iowa7t_variant}" ]; then
      iowa7t_variant="36p"
    fi
    case "${iowa7t_variant}" in
      0p)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P/hipp_outputs}"
        ;;
      32p)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P/hipp_outputs}"
        ;;
      36p)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs}"
        ;;
      noxcpd)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD/hipp_outputs}"
        ;;
      *)
        echo "ERROR: IOWA7T_VARIANT must be 0p, 32p, 36p, or noxcpd, got: ${IOWA7T_VARIANT:-}" >&2
        exit 2
        ;;
    esac
    ;;
  estt)
    if [ -z "${estt_variant}" ]; then
      case "${OUT_ROOT:-}" in
        */DataOrg_PFM_23P/hipp_outputs/cortex_kong2022_areal_mshbm)
          estt_variant="23p"
          ;;
        ""|*/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm)
          estt_variant="36p"
          ;;
        *)
          echo "ERROR: ESTT OUT_ROOT is not an identifiable 23P or 36P root; set ESTT_VARIANT=23p|36p explicitly." >&2
          exit 2
          ;;
      esac
    fi
    case "${estt_variant}" in
      23p)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P/hipp_outputs/cortex_kong2022_areal_mshbm}"
        ;;
      36p)
        out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm}"
        ;;
      *)
        echo "ERROR: ESTT_VARIANT must be 23p or 36p, got: ${ESTT_VARIANT:-}" >&2
        exit 2
        ;;
    esac
    ;;
  iowa_hcp500)
    out_root="${OUT_ROOT:-/old_Users/yojiao/hipp/outputs_iowa_hcp500/inputs/kong2022/cortex_references/routea_areal_mshbm}"
    ;;
  hcp7t)
    out_root="${OUT_ROOT:-${repo_root}/outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm}"
    ;;
  pdc|banda)
    repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
    out_root="${OUT_ROOT:-${repo_root}/outputs_${dataset}/inputs/kong2022/cortex_references/routea_areal_mshbm}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms, stanford, iowa7t, estt, iowa_hcp500, hcp7t, pdc, or banda." >&2
    exit 2
    ;;
esac
if [ "${dataset}" = "iowa7t" ]; then
  case "${iowa7t_branch}" in
    multiband|multi_echo)
      ;;
    *)
      echo "ERROR: DATASET=iowa7t requires IOWA7T_BRANCH=multiband or multi_echo." >&2
      exit 2
      ;;
  esac
elif [ -n "${iowa7t_branch}" ]; then
  echo "ERROR: IOWA7T_BRANCH is only valid with DATASET=iowa7t." >&2
  exit 2
fi
job_id="${1:-}"
subject="${2:-sub-1006}"
subject="sub-${subject#sub-}"
if [ "${dataset}" = "iowa7t" ]; then
  subject_out="${out_root}/subjects/${subject}/${iowa7t_branch}/cortex/areal_mshbm"
else
  subject_out="${out_root}/${subject}"
fi

if [ -z "${job_id}" ]; then
  echo "Usage: $(basename "$0") <job_id> [sub-1006]" >&2
  exit 2
fi

echo "Job: ${job_id}"
echo "Dataset: ${dataset}"
echo "StanfordVariant: ${stanford_variant:-<none>}"
echo "ESTTVariant: ${estt_variant:-<none>}"
echo "Branch: ${iowa7t_branch:-<none>}"
echo "Variant: ${iowa7t_variant:-<none>}"
echo "Subject: ${subject}"
echo "Output: ${subject_out}"
echo

echo "qstat:"
qstat -j "${job_id}" 2>/dev/null || qstat -u yojiao 2>/dev/null || true
echo

echo "qacct:"
qacct -j "${job_id}" 2>/dev/null || true
echo

echo "logs:"
shopt -s nullglob
if [ "${dataset}" = "iowa7t" ]; then
  logs=("${out_root}"/runs/logs/*/*."${job_id}".*.log)
else
  logs=("${out_root}/_sge_logs"/*/*."${job_id}".*.log)
fi
printf '%s\n' "${logs[@]}" | sort
echo

echo "expected label text outputs:"
for path in \
  "${subject_out}/lh_labels.txt" \
  "${subject_out}/rh_labels.txt" \
  "${subject_out}/provenance.json"; do
  if [ -e "${path}" ]; then
    echo "OK      ${path}"
  else
    echo "MISSING ${path}"
  fi
done

if [ "${dataset}" = "estt" ]; then
  echo
  echo "ESTT materialized assets:"
  for path in \
    "${subject_out}/labels/sub-${subject#sub-}_atlas-kong2022_areal_mshbm.L.label.gii" \
    "${subject_out}/labels/sub-${subject#sub-}_atlas-kong2022_areal_mshbm.R.label.gii" \
    "${subject_out}/sub-${subject#sub-}_atlas-kong2022_areal_mshbm.dlabel.nii" \
    "${subject_out}/summary.json" \
    "${subject_out}/stage_manifest.json"; do
    if [ -e "${path}" ]; then
      echo "OK      ${path}"
    else
      echo "MISSING ${path}"
    fi
  done
fi

# Preset data_root contract token: /old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions
