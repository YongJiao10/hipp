# Network-First 166 HPC Bundle

This directory is the clean handoff bundle for the 166-subject HPC run.

It keeps only:

- `network-gradient`
- `network-prob-cluster-nonneg`
- `lynch2024`
- `kong2019`

It does not include:

- any precomputed outputs
- any smoke artifacts
- classic parcel-first parcellation code
- `prob-soft`, `wta`, or `hermosillo2024`
- fallback reads from another local HippoMaps checkout

Start with [docs/HPC_AGENT_HANDOFF.md](/Users/jy/Documents/HippoMaps-network-first/handoff/network_first_166_bundle/docs/HPC_AGENT_HANDOFF.md) and the local [AGENTS.md](/Users/jy/Documents/HippoMaps-network-first/handoff/network_first_166_bundle/AGENTS.md).

Environment bootstrap files are included:

- [environment.yml](/Users/jy/Documents/HippoMaps-network-first/handoff/network_first_166_bundle/environment.yml)
- [setup_hpc_env.sh](/Users/jy/Documents/HippoMaps-network-first/handoff/network_first_166_bundle/setup_hpc_env.sh)
