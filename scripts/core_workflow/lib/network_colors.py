from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"

RGBA = tuple[int, int, int, int]

NETWORK_DISPLAY_NAMES = {
    "Default": "DMN", "Visual": "VN", "Somatomotor": "SMN",
    "DorsalAttention": "DAN", "VentralAttention": "VAN", "Control": "CON",
    "Language": "LAN", "Auditory": "AUD", "Limbic": "LIM",
    "CinguloOpercular": "CO",
    "DefaultA": "DMN-A", "DefaultB": "DMN-B", "DefaultC": "DMN-C",
    "ContA": "CON-A", "ContB": "CON-B", "ContC": "CON-C",
    "SalVenAttnA": "VAN-A", "SalVenAttnB": "VAN-B",
    "DorsAttnA": "DAN-A", "DorsAttnB": "DAN-B",
    "SomMotA": "SMN-A", "SomMotB": "SMN-B",
    "VisualA": "VN-A", "VisualB": "VN-B", "VisualC": "VN-C", "Aud": "AUD",
    "Default_Parietal": "DMN-Par", "Default_Anterolateral": "DMN-AntLat",
    "Default_Dorsolateral": "DMN-DorsLat", "Default_Retrosplenial": "DMN-Retro",
    "Visual_Lateral": "VN-Lat", "Visual_Dorsal/VentralStream": "VN-DV",
    "Visual_V5": "VN-V5", "Visual_V1": "VN-V1", "Frontoparietal": "FPN",
    "Premotor/DorsalAttentionII": "PM/DAN-II", "Salience": "SAL",
    "CinguloOpercular/Action-mode": "CO/AM", "MedialParietal": "MP",
    "Somatomotor_Hand": "SMN-Hand", "Somatomotor_Face": "SMN-Face",
    "Somatomotor_Foot": "SMN-Foot", "SomatoCognitiveAction": "SCAN",
}


def _parse_rgba(name: str, raw: object) -> RGBA:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Network color for {name!r} must be an RGBA list of length 4")
    rgba = tuple(int(v) for v in raw)
    if any(v < 0 or v > 255 for v in rgba):
        raise ValueError(f"Network color for {name!r} has out-of-range RGBA values: {rgba}")
    return rgba  # type: ignore[return-value]


def load_network_colors(path: Path = GLOBAL_NETWORK_STYLE_JSON) -> dict[str, RGBA]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Network style JSON must be an object: {path}")
    out: dict[str, RGBA] = {}
    for key, row in payload.items():
        if not isinstance(row, dict):
            raise ValueError(f"Network style row {key!r} must be an object")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(f"Network style row {key!r} is missing name")
        if name in out:
            raise ValueError(f"Duplicate network color name in {path}: {name}")
        out[name] = _parse_rgba(name, row.get("rgba"))
    return out


def require_network_colors(names: list[str] | tuple[str, ...], path: Path = GLOBAL_NETWORK_STYLE_JSON) -> dict[str, RGBA]:
    colors = load_network_colors(path)
    requested = [str(name) for name in names]
    missing = sorted({name for name in requested if name not in colors})
    if missing:
        raise KeyError(f"Missing network colors in {path}: {', '.join(missing)}")
    return {name: colors[name] for name in requested}


def load_network_hex_colors(path: Path = GLOBAL_NETWORK_STYLE_JSON) -> dict[str, str]:
    return {name: rgba_to_hex(rgba) for name, rgba in load_network_colors(path).items()}


def load_cluster_palette(path: Path = GLOBAL_NETWORK_STYLE_JSON) -> list[RGBA]:
    colors = [(128, 128, 128, 255)]
    colors.extend(load_network_colors(path).values())
    return colors


def rgba_to_hex(rgba: RGBA) -> str:
    r, g, b, _a = rgba
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
