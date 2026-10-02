# Shifted-Cosine Cross-Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one shifted-cosine gradient-spectral algorithm for group and cross-fit analyses, validate HCP cross-fit robustness, then compare HCP and Stanford group results.

**Architecture:** A shared surface-affinity module is consumed by both clustering entrypoints. Manifests record the exact affinity and graph diagnostics; a quantitative gate controls whether isolated group reruns proceed.

**Tech Stack:** Python 3.14, NumPy, SciPy, scikit-learn, nibabel, pytest.

## Global Constraints

- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python tasks.
- Keep native numerical threads at one and total concurrency at or below ten.
- Preserve experimental outputs until the separately approved formalization and archival step.

---

- [x] Add the shared shifted-cosine affinity implementation with tests; the temporary compatibility mode was removed during formalization.
- [x] Route group atlas and cross-fit through the shared implementation and record provenance.
- [x] Add graph diagnostics and a fixed robustness gate with tests.
- [x] Add HCP/Stanford old-versus-new comparison reporting with tests.
- [ ] Run full verification and commit the implementation branch.
- [ ] Run isolated HCP cross-fit and evaluate the gate.
- [ ] If the gate passes, rebuild and compare HCP, Stanford HC, and Stanford Symp group overviews.
- [ ] Derive and render HCP, Stanford HC, and Stanford Symp weighted-fisherz branches from the accepted fixed-K3 shifted-cosine roots.
