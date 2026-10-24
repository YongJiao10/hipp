#!/bin/bash

#$ -N finalize_fmriprep_nih_rtms
#$ -pe smp 1
#$ -q UI,PINC
#$ -j y
#$ -o /Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs

set -euo pipefail

source_job=${1:?Usage: finalize_fmriprep_nih_rtms.sh FMRIPREP_ARRAY_JOB_ID}
[[ "$source_job" =~ ^[0-9]+$ ]] || { echo "ERROR: Invalid source array job ID: $source_job"; exit 1; }
base=/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS
roster=/old_Users/yojiao/hipp/scripts/nih_rtms/sublist.txt
table=/Shared/jianglab/0_scripts/fmriprep_tools/TemplateFlow/tpl-fsaverage/tpl-fsaverage_dseg.tsv
table_sha256=be53e1cd18634bb4091297d5f6ec51e44626c0fa4d57030ebd63a3281fe34582

fail() { echo "ERROR: $*" >&2; exit 1; }
cmp "$roster" "$base/src/sublist.txt" || fail "Local and shared subject rosters differ"
printf '%s  %s\n' "$table_sha256" "$table" | sha256sum --check --status || fail "Canonical FreeSurfer table hash mismatch: $table"

task=0
runs=0
functional=0
declare -A seen=()
while IFS= read -r subject; do
    task=$((task+1))
    [[ "$subject" =~ ^sub-(D[0-9]{2}|HVS[0-9]+)$ ]] || fail "Malformed roster entry at task $task: $subject"
    [[ -z "${seen[$subject]:-}" ]] || fail "Duplicate subject: $subject"
    seen[$subject]=1
    log="$base/src/logs/fmriprep_nih_rtms.o${source_job}.${task}"
    report="$base/derivatives/restfmri/$subject.html"
    grep -Fq 'fMRIPrep finished successfully!' "$log" || fail "No successful scientific workflow receipt: $log"
    grep -Fq '<p>No errors to report!</p>' "$report" || fail "Missing or failed HTML report: $report"
    listing=$(find "$base/bids_input/$subject" -type f -name "${subject}_*_task-rest_bold.nii.gz" | sort)
    if [[ -z "$listing" ]]; then
        [[ "$subject" == sub-D16 ]] || fail "Unexpected missing functional input: $subject"
        continue
    fi
    functional=$((functional+1))
    while IFS= read -r raw; do
        rel=${raw#"$base/bids_input/$subject/"}
        dir=${rel%/*}
        stem=${rel##*/}
        stem=${stem%_bold.nii.gz}
        for suffix in _space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz _space-fsLR_den-91k_bold.dtseries.nii _desc-confounds_timeseries.tsv _desc-confounds_timeseries.json; do
            product="$base/derivatives/restfmri/$subject/$dir/$stem$suffix"
            test -s "$product" || fail "Missing exact fMRIPrep product: $product"
        done
        runs=$((runs+1))
    done <<< "$listing"
done < "$roster"
[[ "$task" == 44 && "$functional" == 43 && "$runs" == 66 ]] || fail "Unexpected cohort counts: subjects=$task functional=$functional runs=$runs"

for filename in desc-aseg_dseg.tsv desc-aparcaseg_dseg.tsv; do
    destination="$base/derivatives/restfmri/$filename"
    if test -e "$destination"; then
        cmp "$table" "$destination" || fail "Existing canonical metadata differs: $destination"
    else
        cp --no-clobber -- "$table" "$destination"
        cmp "$table" "$destination" || fail "Finalized metadata differs: $destination"
    fi
    sha256sum "$destination"
done

printf 'FMRIPREP_FINALIZATION_OK source_array=%s subjects=%d functional_subjects=%d runs=%d canonical_table_sha256=%s\n' "$source_job" "$task" "$functional" "$runs" "$table_sha256"
