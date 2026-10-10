"""Small deterministic geometry and naming helpers shared by the analysis stages."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from animagent.models.anatomy import Side

_SIDE_SUFFIX = re.compile(r"(?P<sep>[._\- ])(?P<side>L|R|l|r|Left|Right|left|right)(?P<rest>(\.\d+)*)$")
_SIDE_PREFIX = re.compile(r"^(?P<side>L|R|Left|Right|left|right)(?P<sep>[._\- ])")
_SWAP = {"L": "R", "R": "L", "l": "r", "r": "l", "Left": "Right", "Right": "Left", "left": "right", "right": "left"}


def side_from_name(name: str) -> Side:
    m = _SIDE_SUFFIX.search(name) or _SIDE_PREFIX.search(name)
    if not m:
        return Side.CENTER
    return Side.LEFT if m.group("side").lower().startswith("l") else Side.RIGHT


def mirror_name(name: str) -> str | None:
    m = _SIDE_SUFFIX.search(name)
    if m:
        return name[: m.start()] + m.group("sep") + _SWAP[m.group("side")] + m.group("rest")
    m = _SIDE_PREFIX.search(name)
    if m:
        return _SWAP[m.group("side")] + m.group("sep") + name[m.end() :]
    return None


def v(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > eps else np.zeros_like(x)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance from p to segment ab and the clamped parameter t in [0, 1]."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-18 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return norm(p - (a + t * ab)), t


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = unit(a), unit(b)
    return float(np.degrees(np.arccos(np.clip(ua @ ub, -1.0, 1.0))))


def closest_local_axis(world_axis: np.ndarray, matrix_world: Sequence[Sequence[float]]) -> tuple[str, float]:
    """Return e.g. ("+X", 0.98): the bone-local axis best aligned with `world_axis`."""
    m = np.asarray(matrix_world, dtype=np.float64)[:3, :3]
    best, best_dot = "+X", -2.0
    for i, label in enumerate("XYZ"):
        axis = unit(m[:, i])
        d = float(axis @ unit(world_axis))
        if abs(d) > best_dot:
            best_dot = abs(d)
            best = ("+" if d >= 0 else "-") + label
    return best, best_dot


def rnd(x: Sequence[float] | np.ndarray, nd: int = 4) -> list[float]:
    return [round(float(c), nd) for c in x]
