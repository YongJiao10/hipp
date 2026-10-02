"""Shared visual style, palettes, and scene configurations for multi-dataset figures."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"


def load_network_hex_colors(path: Path = GLOBAL_NETWORK_STYLE_JSON) -> dict[str, str]:
    """Load network hex colors from global network style json."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for row in payload.values():
        if isinstance(row, dict) and "name" in row and "rgba" in row:
            r, g, b, _a = row["rgba"]
            out[row["name"]] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return out


COLOR_MAP = load_network_hex_colors()

DEFAULT_NET_COLORS: dict[str, str] = {
    "VentralAttention": "#d95f02",
    "Somatomotor": "#1f78b4",
    "Default": "#e31a1c",
    "Auditory": "#18b7c9",
    "DorsalAttention": "#33a02c",
    "Control": "#ff7f00",
    "Visual": "#6a3d9a",
    "Language": "#b15928",
    "CinguloOpercular": "#fb9a99",
}

for k, v in DEFAULT_NET_COLORS.items():
    if k not in COLOR_MAP:
        COLOR_MAP[k] = v

ABBREV_MAP: dict[str, str] = {
    "Default": "DMN",
    "Visual": "VN",
    "Somatomotor": "SMN",
    "DorsalAttention": "DAN",
    "VentralAttention": "VAN",
    "Control": "FPN",
    "Language": "LAN",
    "Auditory": "AUD",
    "CinguloOpercular": "CON",
}


def to_abbrev(net_name: str) -> str:
    """Return standard abbreviation for a functional network."""
    return ABBREV_MAP.get(net_name, net_name[:3].upper())


SUBREGION_NAMES: dict[int, str] = {
    1: "Anterior",
    2: "Middle",
    3: "Posterior",
}

SUBREGION_SHORT_NAMES: dict[int, str] = {
    1: "Ant",
    2: "Mid",
    3: "Post",
}

SUBREGION_COLORS: dict[int, str] = {
    1: "#d95f02",  # Orange for Anterior
    2: "#7570b3",  # Purple for Middle
    3: "#1b9e77",  # Green for Posterior
}
