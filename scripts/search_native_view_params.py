#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


@dataclass
class ViewParams:
    ax: float
    ay: float
    az: float
    vx: float
    vy: float
    vz: float
    theta_deg: float
    vscale: float
    gap: float


def project(points: np.ndarray, p: ViewParams, mirror_left: bool) -> tuple[np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    h = p.ax * x + p.ay * y + p.az * z
    v = p.vx * x + p.vy * y + p.vz * z
    th = np.deg2rad(p.theta_deg)
    c = np.cos(th)
    s = np.sin(th)
    hr = c * h - s * v
    vr = s * h + c * v
    vr = p.vscale * vr
    if mirror_left:
        hr = -hr
    return hr, vr


def normalize_pair(lh: np.ndarray, lv: np.ndarray, rh: np.ndarray, rv: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def norm(h: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = h - np.median(h)
        v = v - np.median(v)
        s = max(np.percentile(np.abs(h), 99), np.percentile(np.abs(v), 99), 1e-6)
        return h / s, v / s

    lh, lv = norm(lh, lv)
    rh, rv = norm(rh, rv)
    lh = lh - gap
    rh = rh + gap
    return lh, lv, rh, rv


def score_layout(lh: np.ndarray, lv: np.ndarray, rh: np.ndarray, rv: np.ndarray) -> float:
    # Targets are chosen from the reference screenshot: tall shape + inward-facing + similar top/bottom.
    def height_width_ratio(h: np.ndarray, v: np.ndarray) -> float:
        w = np.percentile(h, 95) - np.percentile(h, 5)
        hgt = np.percentile(v, 95) - np.percentile(v, 5)
        return float(hgt / max(w, 1e-6))

    l_ratio = height_width_ratio(lh, lv)
    r_ratio = height_width_ratio(rh, rv)

    l_top = np.percentile(lv, 98)
    r_top = np.percentile(rv, 98)
    l_bot = np.percentile(lv, 2)
    r_bot = np.percentile(rv, 2)

    l_inner = np.percentile(lh, 90)
    r_inner = np.percentile(rh, 10)
    center_gap = r_inner - l_inner

    # Prefer narrow top relative to mid for both hemis (the "bean/tall" look).
    def top_narrowness(h: np.ndarray, v: np.ndarray) -> float:
        top = h[v > np.percentile(v, 80)]
        mid = h[(v > np.percentile(v, 35)) & (v < np.percentile(v, 65))]
        if top.size < 20 or mid.size < 20:
            return 0.0
        return float((np.percentile(mid, 95) - np.percentile(mid, 5)) - (np.percentile(top, 95) - np.percentile(top, 5)))

    narrow = top_narrowness(lh, lv) + top_narrowness(rh, rv)

    # Objective (smaller is better)
    obj = 0.0
    obj += (l_ratio - 2.15) ** 2
    obj += (r_ratio - 2.15) ** 2
    obj += 1.2 * (l_top - r_top) ** 2
    obj += 0.8 * (l_bot - r_bot) ** 2
    obj += 0.9 * (center_gap - 1.65) ** 2
    obj += 0.5 * (max(0.0, 0.20 - narrow)) ** 2
    return float(obj)


def render_preview(
    out_png: Path,
    left_xy: tuple[np.ndarray, np.ndarray],
    right_xy: tuple[np.ndarray, np.ndarray],
    left_tri: np.ndarray,
    right_tri: np.ndarray,
    left_lab: np.ndarray,
    right_lab: np.ndarray,
    style_json: Path,
    title: str,
) -> None:
    style_raw = json.loads(style_json.read_text(encoding="utf-8"))
    style = {int(k): np.array(v["rgba"], dtype=np.float32) / 255.0 for k, v in style_raw.items()}
    style.setdefault(0, np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32))
    max_key = max(style)
    colors = np.zeros((max_key + 1, 4), dtype=np.float32)
    for k, rgba in style.items():
        colors[k] = rgba
    cmap = ListedColormap(colors)

    lx, ly = left_xy
    rx, ry = right_xy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=250)
    ax.tripcolor(lx, ly, left_tri, left_lab, cmap=cmap, shading="flat", vmin=0, vmax=max_key)
    ax.tripcolor(rx, ry, right_tri, right_lab, cmap=cmap, shading="flat", vmin=0, vmax=max_key)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search native view params with 1000 random trials.")
    parser.add_argument("--left-surface", required=True)
    parser.add_argument("--right-surface", required=True)
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--style-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-preview", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    left_surf = nib.load(args.left_surface)
    right_surf = nib.load(args.right_surface)
    lpts = left_surf.agg_data("pointset")
    rpts = right_surf.agg_data("pointset")
    ltri = left_surf.agg_data("triangle")
    rtri = right_surf.agg_data("triangle")
    llab = np.load(args.left_labels).astype(np.int32)
    rlab = np.load(args.right_labels).astype(np.int32)

    lpts = lpts - np.median(lpts, axis=0, keepdims=True)
    rpts = rpts - np.median(rpts, axis=0, keepdims=True)

    best = None
    best_score = float("inf")
    best_xy = None
    for _ in range(args.trials):
        p = ViewParams(
            ax=1.0,
            ay=float(rng.uniform(0.15, 0.85)),
            az=float(rng.uniform(-0.20, 0.20)),
            vx=float(rng.uniform(-0.10, 0.10)),
            vy=float(rng.uniform(-0.20, 0.30)),
            vz=1.0,
            theta_deg=float(rng.uniform(-35.0, 35.0)),
            vscale=float(rng.uniform(1.05, 1.65)),
            gap=float(rng.uniform(0.95, 1.45)),
        )
        lh, lv = project(lpts, p, mirror_left=True)
        rh, rv = project(rpts, p, mirror_left=False)
        lh, lv, rh, rv = normalize_pair(lh, lv, rh, rv, p.gap)
        sc = score_layout(lh, lv, rh, rv)
        if sc < best_score:
            best_score = sc
            best = p
            best_xy = (lh, lv, rh, rv)

    assert best is not None and best_xy is not None
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "score": best_score,
                "trials": args.trials,
                "params": {
                    "ax": best.ax,
                    "ay": best.ay,
                    "az": best.az,
                    "vx": best.vx,
                    "vy": best.vy,
                    "vz": best.vz,
                    "theta_deg": best.theta_deg,
                    "vscale": best.vscale,
                    "gap": best.gap,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lh, lv, rh, rv = best_xy
    render_preview(
        Path(args.out_preview),
        (lh, lv),
        (rh, rv),
        ltri,
        rtri,
        llab,
        rlab,
        Path(args.style_json),
        f"Best of {args.trials} trials, score={best_score:.4f}",
    )
    print(out_json)
    print(args.out_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
