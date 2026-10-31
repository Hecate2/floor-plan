"""底层几何：读图、去斜、墙体掩码、矩形分解、墙段合并与吸附。"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import os
import numpy as np


def imread_any(path: str) -> np.ndarray:
    """支持 webp/png/jpg；中文路径与 webp 都走二进制解码。"""
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图像: {path}")
    return img


def estimate_skew(wall_mask: np.ndarray) -> float:
    """用长直墙的 Hough 方向估计图纸倾斜角（度，逆时针为正）。"""
    edges = cv2.Canny((wall_mask * 255).astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=30,
                            minLineLength=50, maxLineGap=4)
    if lines is None:
        return 0.0
    angs, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) % 90.0
        if a > 45:
            a -= 90.0
        angs.append(a)
        weights.append(float(np.hypot(x2 - x1, y2 - y1)))
    angs, weights = np.array(angs), np.array(weights)
    # 加权中位数，抗少量斜墙干扰
    order = np.argsort(angs)
    cum = np.cumsum(weights[order])
    med = angs[order][np.searchsorted(cum, cum[-1] / 2)]
    # 只用靠近中位数的样本重算均值
    sel = np.abs(angs - med) < 1.5
    if sel.sum() == 0:
        return float(med)
    return float(np.average(angs[sel], weights=weights[sel]))


def wall_and_line_masks(gray: np.ndarray, p) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (wall_mask, line_mask)。wall=厚实深色墙体，line=所有深色线（含窗/门弧/文字）。"""
    dark = (gray < p.wall_gray).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.wall_open_ksize,) * 2)
    wall = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    keep = np.zeros_like(wall)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= p.min_wall_area:
            keep[lab == i] = 1
    lines = (gray < p.line_gray).astype(np.uint8)
    return keep, lines


def decompose_rects(mask: np.ndarray, min_len: int = 12) -> List[Tuple[int, int, int, int]]:
    """把正交墙体掩码拆成一组轴对齐矩形 (x0, y0, x1, y1)。
    思路：找最长水平行程 -> 沿该行程上下生长 -> 挖掉 -> 重复。
    对角/斜墙会退化成若干小矩形，由后续墙段合并处理。"""
    # 大图先在降采样掩码上分解再换算回原尺寸：逐行扫描的代价与行数成正比
    f = max(1, int(np.ceil(max(mask.shape) / 1600.0)))
    if f > 1:
        small = cv2.resize((mask > 0).astype(np.uint8), (mask.shape[1] // f, mask.shape[0] // f),
                           interpolation=cv2.INTER_AREA)
        rects = decompose_rects((small >= 1).astype(np.uint8), max(4, min_len // f))
        return [(x0 * f, y0 * f, x1 * f, y1 * f) for (x0, y0, x1, y1) in rects]

    m = (mask > 0).astype(np.uint8).copy()
    rects: List[Tuple[int, int, int, int]] = []
    H, W = m.shape

    def longest_runs(arr):
        """逐行找 1 的最长行程，返回 (max_len, line_idx, s0, s1)。"""
        best = (0, -1, 0, 0)
        for i in range(arr.shape[0]):
            row = arr[i]
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row, [0])))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            if starts.size == 0:
                continue
            ln = ends - starts
            k = int(ln.argmax())
            if int(ln[k]) > best[0]:
                best = (int(ln[k]), i, int(starts[k]), int(ends[k]))
        return best

    while True:
        bh = longest_runs(m)          # 水平行程
        bv = longest_runs(m.T)        # 垂直行程
        if max(bh[0], bv[0]) < min_len:
            break
        if bh[0] >= bv[0]:
            # 沿 y 方向生长
            ln, y, x0, x1 = bh
            y0, y1 = y, y + 1
            while y0 - 1 >= 0 and m[y0 - 1, x0:x1].mean() >= 0.85:
                y0 -= 1
            while y1 < H and m[y1, x0:x1].mean() >= 0.85:
                y1 += 1
            while x0 - 1 >= 0 and m[y0:y1, x0 - 1].mean() >= 0.85:
                x0 -= 1
            while x1 < W and m[y0:y1, x1].mean() >= 0.85:
                x1 += 1
        else:
            # 沿 x 方向生长
            ln, x, y0, y1 = bv
            x0, x1 = x, x + 1
            while x0 - 1 >= 0 and m[y0:y1, x0 - 1].mean() >= 0.85:
                x0 -= 1
            while x1 < W and m[y0:y1, x1].mean() >= 0.85:
                x1 += 1
            while y0 - 1 >= 0 and m[y0 - 1, x0:x1].mean() >= 0.85:
                y0 -= 1
            while y1 < H and m[y1, x0:x1].mean() >= 0.85:
                y1 += 1
        if (x1 - x0) < min_len and (y1 - y0) < min_len:
            m[y0:y1, x0:x1] = 0
            continue
        rects.append((x0, y0, x1, y1))
        m[y0:y1, x0:x1] = 0
    return rects


@dataclass
class Seg:
    """轴对齐墙段：orient='h' 时 line=y, 区间沿 x；'v' 时 line=x, 区间沿 y。"""
    orient: str
    line: float
    s0: float
    s1: float
    t: float          # 厚度
    e0: int = 0       # 端点吸附后的整型坐标
    e1: int = 0

    @property
    def length(self):
        return self.s1 - self.s0

    def endpoints(self):
        if self.orient == "h":
            return (self.e0, self.line), (self.e1, self.line)
        return (self.line, self.e0), (self.line, self.e1)


def rects_to_segs(rects, p) -> List[Seg]:
    """矩形 -> 墙段（长边为走向，短边为厚度），并做共线合并 + 端点吸附。"""
    segs: List[Seg] = []
    for (x0, y0, x1, y1) in rects:
        w, h = x1 - x0, y1 - y0
        if w >= h:
            segs.append(Seg("h", (y0 + y1) / 2, x0, x1, h))
        else:
            segs.append(Seg("v", (x0 + x1) / 2, y0, y1, w))

    # 形状过滤：墙必须“细长且有一定厚度”，否则是门弧/箭头/家具碎块
    def plausible(s):
        return (s.t >= 3.0 and s.length >= max(p.min_wall_len, 1.8 * s.t))

    segs = [s for s in segs if plausible(s)]
    if not segs:
        return []
    med_t = float(np.median([s.t for s in segs]))
    # 太薄的是门弧/家具轮廓，太厚的是涂黑的家具/设备块
    segs = [s for s in segs
            if 0.45 * med_t <= s.t <= max(2.6 * med_t, med_t + 8)]

    # 共线合并：同方向、中心线接近、区间相邻
    merged = True
    while merged:
        merged = False
        out: List[Seg] = []
        used = [False] * len(segs)
        for i, a in enumerate(segs):
            if used[i]:
                continue
            cur = a
            for j in range(i + 1, len(segs)):
                b = segs[j]
                if used[j] or b.orient != cur.orient:
                    continue
                if abs(b.line - cur.line) > max(2.0, 0.6 * max(a.t, b.t)):
                    continue
                if b.t and abs(b.t - cur.t) > 0.5 * max(a.t, b.t):
                    continue
                if b.s0 > cur.s1 + p.merge_gap_px or b.s1 < cur.s0 - p.merge_gap_px:
                    continue
                cur = Seg(cur.orient, (cur.line * cur.length + b.line * b.length) / max(cur.length + b.length, 1e-6),
                          min(cur.s0, b.s0), max(cur.s1, b.s1), max(cur.t, b.t))
                used[j] = True
                merged = True
            out.append(cur)
        segs = out

    # 轴网吸附：把同方向端点聚类到公共坐标
    for orient in ("h", "v"):
        vals = []
        for s in segs:
            if s.orient == orient:
                vals += [s.s0, s.s1]
        if not vals:
            continue
        vals = np.array(sorted(vals))
        clusters = []
        for v in vals:
            if clusters and v - clusters[-1][-1] <= p.snap_tol_px:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        centers = [float(np.mean(c)) for c in clusters]
        def snap(v):
            i = int(np.argmin([abs(v - c) for c in centers]))
            return int(round(centers[i]))
        for s in segs:
            if s.orient == orient:
                s.e0, s.e1 = snap(s.s0), snap(s.s1)
                if s.e0 > s.e1:
                    s.e0, s.e1 = s.e1, s.e0
    for s in segs:
        s.line = float(int(round(s.line)))
    return [s for s in segs if s.length >= p.min_wall_len]


def rasterize_segs(segs: List[Seg], shape) -> np.ndarray:
    """把墙段回描成掩码（以矢量化结果为准，避免把深色家具当成墙）。"""
    out = np.zeros(shape, np.uint8)
    for s in segs:
        t = max(1, int(round(s.t)))
        if s.orient == "h":
            y = int(round(s.line - t / 2))
            cv2.rectangle(out, (int(s.e0), y), (int(s.e1), y + t), 1, -1)
        else:
            x = int(round(s.line - t / 2))
            cv2.rectangle(out, (x, int(s.e0)), (x + t, int(s.e1)), 1, -1)
    return out


def split_thin_ink(ink: np.ndarray, wall: np.ndarray, thick: float = 3.0) -> np.ndarray:
    """区分"线"与"实心色块"：实心的深色家具不该被当成墙/屏障。
    返回只保留细线的墨迹掩码。"""
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    solid = (dt >= thick).astype(np.uint8)
    solid = cv2.dilate(solid, np.ones((5, 5), np.uint8))
    wall_d = cv2.dilate((wall > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    drop = (solid > 0) & (wall_d == 0)
    return ((ink > 0) & ~drop).astype(np.uint8)
