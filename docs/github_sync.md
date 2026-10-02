# GitHub scope and synchronization

GitHub `main` contains source code, documentation, configuration, environment
files, existing workflow manifests and tests. Generated results are kept on
the local filesystem: output and presentation trees, result CSV/TSV/JSON files,
surface arrays, rendered images, caches, backups and archives are excluded.
Small diagram input images under `scripts/visualize/flowchart/compact/assets/`
and Workbench scene configurations remain tracked because they are inputs.

The root `.gitignore` uses explicit source-file and configuration exceptions.
A new generated file is ignored unless an exception deliberately includes it.
Before tracking a new binary or configuration input, confirm that it is a
required input and add a specific exception if needed. Do not force-add result
files or output directories. Existing source files can be updated normally.

## Source-history replacement

On 2026-10-02, the repository owner authorized replacing GitHub `main` with a
source-only root commit. The earlier local and remote histories had no common
ancestor. The complete prior local `main` history remains accessible through
`local/main-before-github-source-sync`; existing local data and outputs remain
on disk. Other local and remote branches are outside this replacement.

Local `main` follows the new GitHub source history. Future source updates use
ordinary commits and pushes; do not merge the full-history backup into `main`,
since that would reintroduce generated files and their historical blobs.

```bash
git status --short
git add -- path/to/changed_source.py path/to/updated_document.md
git diff --cached --stat
git commit -m "Describe the source or documentation update"
git push origin main
```

A fresh clone does not include local datasets or derived products. Configure
the external data paths and runtime environments using the workflow runbooks
before executing analysis. Code and documentation changes that affect workflow
behavior must be committed together, as required by `AGENTS.md`.
