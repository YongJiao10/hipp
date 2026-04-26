# T1-only HippUnfold Handoff (for next agent)

## 0) Current progress (updated 2026-04-24 ~02:34 UTC)

- **132/166 done (80%)**, 4 running, 30 incomplete
- Manager running: timestamp `20260423T061540Z`, concurrency=4
- All recent subjects reporting `rc=0` — manifest fix confirmed working (see §10)

## 1) Read this first
- Project root: `/home/yojiao/hpchome/hipp`
- Current target run is **T1-only** (not T2-only, not T2+T1reg).
- Before doing anything else, run:
  - `./check_t1only_progress.sh`
- That script is the canonical quick status check for this handoff.

## 2) Current run configuration
- Batch root:
  - `outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch`
- Runner:
  - `scripts/run_hippunfold_local_t1w_isolated_cache.sh`
- Manager launcher:
  - `scripts/run_remaining_hippunfold_background.sh --worker ... --concurrency 4`
- Parallel cap rule in repo: max 10; active strategy here is **concurrency=4**.

## 3) Historical context (important to avoid confusion)
- Earliest run path was T2-only.
- Then an independent `T2w + --t1-reg-template` run was started but not completed.
- That `T2w+T1reg` result directory was deleted in an interrupted cleanup sequence; treat current on-disk result count there as 0.
- Latest active plan is **independent T1-only**.

## 4) Issues already encountered and fixed
- Problem A: T1 model tar in shared asset store was corrupted (small file, tar invalid).
  - Fixed by re-downloading official Task101 model tar from HippUnfold config URL.
- Problem B: `unpack_nnunet_model` race when 4 subjects unpack to same target simultaneously.
  - Fixed by pre-unpacking Task101 model once (serial) into shared asset store.
- Problem C: detached nohup manager got reaped in this environment.
  - Workaround: run manager in a persistent interactive session (`--worker`) and poll from separate shell.

## 5) Shared asset store locations
- Persistent reusable assets:
  - `runtime/hippunfold_asset_store_shared/hippunfold_assets`
  - `runtime/hippunfold_asset_store_shared/runtime_source_assets`
- Pre-unpacked T1 model target dir:
  - `runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best`

## 6) Progress / monitoring commands
- Primary one-liner:
  - `./check_t1only_progress.sh`
- Optional explicit batch root:
  - `./check_t1only_progress.sh /home/yojiao/hpchome/hipp/outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch`
- Live manager tail (latest):
  - `tail -f outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch/_batch_logs/manager_remaining_*_conc4.log`

## 7) Relaunch command (if manager is not running)
Use exactly this pattern:

```bash
cd /home/yojiao/hpchome/hipp
scripts/run_remaining_hippunfold_background.sh \
  --worker \
  --timestamp "$(date -u +%Y%m%dT%H%M%SZ)" \
  --concurrency 4 \
  --batch-root outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch \
  --runner scripts/run_hippunfold_local_t1w_isolated_cache.sh
```

## 8) Guardrails for next agent
- Do not mix old T2/T2+T1reg outputs with T1-only outputs.
- Keep using shared asset store; do not rename it to cache in docs.
- Avoid deleting any existing result directories unless user explicitly confirms exact path list.
- If run behavior changes (paths/runner/env/workflow), update docs in same change.

## 9) Pitfalls already hit (do not repeat)
- **Duplicate manager queues can silently stack.**
  - Symptom: multiple timestamps (`...045500Z`, `...051230Z`, `...`) alive together, effective concurrency blows past 4.
  - Fix pattern: kill all `run_remaining_hippunfold_background.sh --worker` first, verify zero, then start exactly one timestamp.
- **Progress script can under-report running=0 if path matching is too strict.**
  - Root cause: some commands contain relative batch path, not absolute.
  - Current `check_t1only_progress.sh` matches both absolute and repo-relative batch path; keep this behavior.
- **Task101 T1 model tar corruption happened once.**
  - Symptom: `unpack_nnunet_model` fails with tar errors (`not a tar archive`).
  - Current state: re-downloaded and validated (`tar -tf` OK) in shared asset store.
- **Parallel unpack race on Task101 model directory happened once.**
  - Symptom: tar extract errors like `Cannot open: File exists` / mkdir race.
  - Current workaround: pre-unpack Task101 model once serially before bulk rerun.
- **Detached/nohup behavior can be unreliable in this session environment.**
  - If detached manager appears to “start then disappear”, run `--worker` in a persistent session and poll from another shell.

## 10) Manifest cache fix (2026-04-23)

- **Problem:** `_batch_logs/output_integrity_manifest.json` was built from an old T2w run (`outputs_migration/.../sub-100610`). It expected `space-T2w` surface files that T1-only runs never produce → every subject reported `rc=1 (not complete after 3 attempts)`, triggering 2 wasted retries each.
- **Fix applied:** Deleted stale manifest. Integrity checker auto-rebuilt it from a successful T1-only log (`sub-200210`), now correctly expects `space-T1w` files.
- **Verification:** `is_subject_complete()` returns 0 for completed subjects; manager logs `rc=0` from this point forward.
- **If manifest corruption recurs:** Delete `_batch_logs/output_integrity_manifest.json` — it will be rebuilt on next `is_subject_complete()` call.
