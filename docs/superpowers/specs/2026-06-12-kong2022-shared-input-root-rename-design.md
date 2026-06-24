# Kong2022 Shared Input Root Rename Design

## Goal

Remove the HCP-cohort-specific `_151` suffix from the canonical RouteA and
surface shared-input roots so the same defaults describe HCP, Iowa, Stanford,
and other datasets correctly.

## Canonical Layout

```text
shared_inputs/
  routea/
  surface/
```

`resolve_kong2022_routea_root()` returns `shared_inputs/routea` and
`resolve_kong2022_surface_root()` returns `shared_inputs/surface` for every
dataset key.

## Scope

This change renames only the base RouteA reference/FC root and the base surface
root. The independent experimental or alternate-route roots
`routea_ctxreg_151`, `routea_ctxdrop_151`, and `routeb_151` retain their current
names because no matching migration was requested and changing them would alter
separate workflow branches.

Maintained code and documentation must stop treating `routea_151` and
`surface_151` as canonical defaults. Historical/provenance documents may retain
old paths only when explicitly describing retired outputs.

## Iowa7T Migration

Move `sub-04172026` and all contained reference, FC, surface, mask, and time
series files from `routea_151` and `surface_151` into the existing empty
`routea` and `surface` directories. Do not copy data.

Refresh the individual workflow outputs with explicit new input roots so combo
manifests point to the canonical paths. Rebuild overview products to prove the
downstream reader follows the new layout.
