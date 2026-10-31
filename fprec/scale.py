"""比例尺标定：从尺寸标注链 / 图上标注面积 反推 mm-per-pixel。

原理：标注链上一个数字写在自己那一段的正中间，于是相邻两个数字的中心距
≈ (w_i + w_{i+1}) / 2 / s，对多对取鲁棒中位数即可，不需要检测刻度线。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .text import Token


@dataclass
class ScaleResult:
    mm_per_px: float
    method: str            # dimension_chain | printed_area | hint | default_door
    confidence: float
    samples: List[float]
    detail: str = ""


_INT_RE = re.compile(r"^\d{3,5}$")
_AREA_RE = re.compile(r"^(\d{1,3})(?:[.,](\d))?\s*(?:m|M|m²|m2|㎡)?$")


def _is_chain_number(t: Token) -> bool:
    txt = t.text.strip().replace(" ", "")
    if not _INT_RE.match(txt):
        return False
    v = int(txt)
    return 300 <= v <= 20000


def find_chains(tokens: Sequence[Token], tol: int = 12, min_len: int = 3) -> List[List[Token]]:
    """把数字 token 聚成横向/纵向两条标注链（按行/列聚类）。"""
    vals = [t for t in tokens if _is_chain_number(t)]
    if len(vals) < min_len:
        return []
    chains: List[List[Token]] = []

    # 横向链：按 cy 聚类
    vals_h = sorted(vals, key=lambda t: t.cy)
    rows: List[List[Token]] = []
    for t in vals_h:
        if rows and abs(t.cy - np.mean([r.cy for r in rows[-1]])) <= tol:
            rows[-1].append(t)
        else:
            rows.append([t])
    for r in rows:
        if len(r) >= min_len:
            chains.append(sorted(r, key=lambda t: t.cx))

    # 纵向链：按 cx 聚类（竖排标注，OCR 前已把图转正）
    vals_v = sorted(vals, key=lambda t: t.cx)
    cols: List[List[Token]] = []
    for t in vals_v:
        if cols and abs(t.cx - np.mean([c.cx for c in cols[-1]])) <= tol:
            cols[-1].append(t)
        else:
            cols.append([t])
    for c in cols:
        if len(c) >= min_len:
            chains.append(sorted(c, key=lambda t: t.cy))
    return chains


def scale_from_chains(tokens: Sequence[Token]) -> Optional[ScaleResult]:
    samples: List[float] = []
    used = 0
    for chain in find_chains(list(tokens)):
        pts = [(t.cx, t.cy) for t in chain]
        vals = [int(t.text.strip()) for t in chain]
        for i in range(len(chain) - 1):
            d = float(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
            if d < 12:
                continue
            s = (vals[i] + vals[i + 1]) / (2.0 * d)
            if 2.0 < s < 200.0:
                samples.append(s)
        used += 1
    if len(samples) < 2:
        return None
    arr = np.array(samples)
    med = float(np.median(arr))
    keep = arr[np.abs(arr - med) / med < 0.15]
    if keep.size == 0:
        return None
    s = float(np.median(keep))
    spread = float(np.median(np.abs(keep - s)) / s)
    conf = float(np.clip(1.0 - 2.5 * spread, 0.2, 0.95))
    return ScaleResult(s, "dimension_chain", round(conf, 2), samples,
                       detail=f"{used} 条标注链，{len(samples)} 个相邻对，离散度 {spread:.3f}")


def scale_from_printed_areas(pairs: Sequence[Tuple[float, float]]) -> Optional[ScaleResult]:
    """pairs = [(标注面积 m², 该房间像素面积), ...] -> 最小二乘/中位数尺度。"""
    if len(pairs) < 2:
        return None
    samples = []
    for m2, px in pairs:
        if m2 <= 0 or px <= 0:
            continue
        samples.append(float(np.sqrt(m2 * 1e6 / px)))     # mm/px
    if len(samples) < 2:
        return None
    arr = np.array(samples)
    med = float(np.median(arr))
    keep = arr[np.abs(arr - med) / med < 0.2]
    s = float(np.median(keep))
    spread = float(np.median(np.abs(keep - s)) / s)
    conf = float(np.clip(1.0 - 2.0 * spread, 0.2, 0.85))
    return ScaleResult(s, "printed_area", round(conf, 2), samples,
                       detail=f"{len(keep)}/{len(samples)} 个房间面积参与拟合，离散度 {spread:.3f}")


def printed_area_tokens(tokens: Sequence[Token], min_m2=1.0, max_m2=300.0):
    """从 OCR token 里挑出形如 11.2 / 3.9m² 的面积标注。"""
    out = []
    for t in tokens:
        txt = t.text.strip().replace(" ", "")
        m = _AREA_RE.match(txt)
        if not m:
            continue
        whole, frac = m.group(1), m.group(2)
        val = float(whole + ("." + frac if frac else ""))
        if not (min_m2 <= val <= max_m2):
            continue
        out.append((val, t))
    return out
