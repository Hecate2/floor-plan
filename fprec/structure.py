"""结构推理：洞口（门/窗）、建筑轮廓、房间分区与多边形。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import Seg


# ============================================================ 洞口

@dataclass
class Opening:
    type: str                 # door | window | opening
    orient: str               # h | v
    line: int                 # 墙中心线坐标(px)
    s0: int                   # 洞口起点
    s1: int                   # 洞口终点
    thickness: float
    confidence: float = 0.7

    @property
    def width(self):
        return self.s1 - self.s0

    @property
    def center(self) -> Tuple[float, float]:
        mid = (self.s0 + self.s1) / 2
        return (mid, self.line) if self.orient == "h" else (self.line, mid)

    def rect(self) -> Tuple[int, int, int, int]:
        t = max(self.thickness, 4) / 2
        if self.orient == "h":
            return (self.s0, int(round(self.line - t)), self.s1, int(round(self.line + t)) + 1)
        return (int(round(self.line - t)), self.s0, int(round(self.line + t)) + 1, self.s1)


def candidate_gaps(segs: List[Seg], p) -> List[Opening]:
    """同一墙线上相邻墙段之间的空隙 => 候选洞口（尚未区分门/窗）。"""
    ops: List[Opening] = []
    groups: Dict[Tuple[str, int], List[Seg]] = {}
    # 中心线聚类，容忍 4px 的抖动
    for s in segs:
        key = None
        for (o, line) in groups:
            if o == s.orient and abs(line - s.line) <= 4:
                key = (o, line)
                break
        groups.setdefault(key or (s.orient, int(round(s.line))), []).append(s)
    for (orient, line), group in groups.items():
        group.sort(key=lambda s: s.e0)
        for a, b in zip(group, group[1:]):
            gap = b.e0 - a.e1
            if gap < p.min_opening_px or gap > p.max_opening_px:
                continue
            if abs(a.t - b.t) > max(4.0, 0.6 * max(a.t, b.t)):
                continue
            ops.append(Opening("opening", orient, line, int(a.e1), int(b.e0),
                               float((a.t + b.t) / 2), 0.3))
    return ops


def classify_openings(ops: List[Opening], fp: np.ndarray, ink: np.ndarray,
                      p, mm_per_px: Optional[float] = None) -> List[Opening]:
    """按 临外墙 / 洞宽 / 符号特征 区分 窗、门、开敞洞口。"""
    edge = cv2.morphologyEx(fp, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    out: List[Opening] = []
    for o in ops:
        mx = (o.s0 + o.s1) // 2
        px, py = (mx, o.line) if o.orient == "h" else (o.line, mx)
        if not (0 <= py < fp.shape[0] and 0 <= px < fp.shape[1]) or fp[py, px] == 0:
            continue
        on_edge = edge[max(0, py - 4):py + 5, max(0, px - 4):px + 5].sum() > 0
        w_mm = o.width * (mm_per_px or 0.0)
        if w_mm and w_mm < 350:
            continue                     # 太窄，是墙体缺口而非门窗
        wide = (w_mm > p.max_opening_mm) if mm_per_px else (o.width > 90)
        # 沿墙线取一条 5px 高的带子：窗有玻璃线贯穿（密度高），门洞是空的（密度低）
        if o.orient == "h":
            band = ink[max(0, o.line - 2):o.line + 3, o.s0:o.s1]
        else:
            band = ink[o.s0:o.s1, max(0, o.line - 2):o.line + 3]
        density = float(band.mean()) if band.size else 0.0
        if wide:
            typ, conf = "opening", 0.55
        elif density > 0.28:
            typ, conf = "window", 0.85 if on_edge else 0.7
        else:
            typ, conf = "door", 0.85 if not on_edge else 0.7
        if typ == "window" and on_edge:
            conf = min(0.95, conf + 0.05)
        o.type, o.confidence = typ, round(conf, 2)
        out.append(o)
    return out


# ============================================================ 轮廓

def add_bridges(mask: np.ndarray, ops, pad: int = 4) -> np.ndarray:
    """把洞口封堵画进掩码（外轮廓/种子阶段共用）。"""
    out = (mask > 0).astype(np.uint8).copy()
    for o in ops or []:
        x0, y0, x1, y1 = o.rect()
        out[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 1
    return out


def footprint(ink: np.ndarray, grow: int = 10) -> np.ndarray:
    """建筑外轮廓（含阳台）：墨迹膨胀成整体 -> 取最大块 -> 填洞 -> 腐蚀还原。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    big = cv2.dilate((ink > 0).astype(np.uint8), k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(big, 8)
    if n <= 1:
        return np.zeros_like(ink, np.uint8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(ink.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return cv2.erode(out, k)


# ============================================================ 房间

@dataclass
class RoomBlob:
    mask: np.ndarray
    polygon: np.ndarray      # Nx2 px
    area_px: float
    bbox: Tuple[int, int, int, int]
    label: Optional[str] = None
    area_printed: Optional[float] = None


def _grow(lab: np.ndarray, free: np.ndarray, iters: int) -> np.ndarray:
    """逐像素测地增长的原语（一次一圈）。"""
    k = np.ones((3, 3), np.uint8)
    lab = lab.astype(np.float32)
    free_b = free > 0
    for _ in range(int(iters)):
        dil = cv2.dilate(lab, k)
        grow = (lab == 0) & (dil > 0) & free_b
        if not grow.any():
            break
        lab[grow] = dil[grow]
    return lab


def _propagate(lab: np.ndarray, free: np.ndarray, iters: int = 300) -> np.ndarray:
    """种子标签在自由空间内测地漫延（不能穿过任何线）。

    大图上先降采样生长、再全分辨率精修——纯逐像素迭代在 3000px 图上要几十秒。
    """
    h, w = lab.shape
    f = 4 if max(h, w) > 1500 else (2 if max(h, w) > 800 else 1)
    if f == 1:
        return _grow(lab, free, iters)
    sw, sh = max(1, w // f), max(1, h // f)
    small_free = cv2.resize(cv2.erode((free > 0).astype(np.uint8), np.ones((f, f), np.uint8)),
                            (sw, sh), interpolation=cv2.INTER_NEAREST)
    sl = cv2.resize(lab.astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST)
    sl[small_free == 0] = 0
    sl = _grow(sl, small_free, max(60, int(iters * 3 / f)))
    lab = cv2.resize(sl, (w, h), interpolation=cv2.INTER_NEAREST)
    lab[(free == 0)] = 0
    return _grow(lab, free, max(20, f * 8))


def orthogonalize(poly: np.ndarray, snap_deg: float) -> np.ndarray:
    """把接近水平/垂直的边吸附成正交；其余（斜墙、弧形阳台）保持原样。"""
    pts = poly.reshape(-1, 2).astype(float)
    n = len(pts)
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            dx, dy = pts[j] - pts[i]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            ang = abs(np.degrees(np.arctan2(dy, dx))) % 180
            if ang < snap_deg or ang > 180 - snap_deg:
                pts[j, 1] = pts[i, 1]
            elif abs(ang - 90) < snap_deg:
                pts[j, 0] = pts[i, 0]
    return pts


def estimate_door_px(segs: List[Seg]) -> float:
    """用墙厚估计门的像素宽度：门宽 ≈ 4.5 倍墙厚（200mm 墙 -> 900mm 门）。"""
    if not segs:
        return 30.0
    t = float(np.median([s.t for s in segs]))
    return float(np.clip(4.5 * t, 18, 90))


def _absorb_unassigned(lab: np.ndarray, free: np.ndarray, passes: int = 3) -> np.ndarray:
    """把没分配到的自由像素（家具/洁具内部等封闭小块）归给最近的主导房间。

    逐个连通域在 bbox 窗口里做，避免在整图上反复膨胀（3000px 大图会慢十倍）。
    """
    k3 = np.ones((3, 3), np.uint8)
    H, W = lab.shape
    for _ in range(passes):
        un = ((lab == 0) & (free > 0)).astype(np.uint8)
        if not un.any():
            break
        n, ul, st, _ = cv2.connectedComponentsWithStats(un, 8)
        for i in range(1, n):
            x, y, w, h = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], \
                st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            pad = 4
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            reg = (ul[y0:y1, x0:x1] == i)
            ring = (cv2.dilate(reg.astype(np.uint8), k3, iterations=3) > 0) & ~reg
            vals = lab[y0:y1, x0:x1][ring]
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            ids, cnt = np.unique(vals, return_counts=True)
            if cnt.max() / cnt.sum() < 0.6:
                continue
            lab[y0:y1, x0:x1][reg] = ids[cnt.argmax()]
    return lab


def solid_blocks(ink: np.ndarray, wall_vec: np.ndarray, min_area: int = 80):
    """深色实心块（衣柜、床、洁具等）：看得见但不是墙，其占地应计入房间面积。"""
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    solid = (dt >= 3.0).astype(np.uint8)
    solid = cv2.dilate(solid, np.ones((7, 7), np.uint8))
    solid &= ((wall_vec > 0).astype(np.uint8) == 0).astype(np.uint8)
    n, sl, st, _ = cv2.connectedComponentsWithStats(solid, 8)
    out = []
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        ys, xs = np.where(sl == i)
        out.append((ys, xs))
    return out


def _wall_contact(mask: np.ndarray, wall_d: np.ndarray) -> float:
    """区域边界上落在墙体上的比例——用来判定"这是不是被墙围出来的房间"。"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0
    pad = 2
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    m = mask[y0:y1, x0:x1] > 0
    ring = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~m
    tot = int(ring.sum())
    if tot == 0:
        return 0.0
    return float((ring & (wall_d[y0:y1, x0:x1] > 0)).sum()) / tot


def segment_rooms(wall: np.ndarray, ink: np.ndarray, p,
                  seed_radius: Optional[int] = None,
                  segs: Optional[List[Seg]] = None,
                  bridges: Optional[List["Opening"]] = None,
                  wall_vec: Optional[np.ndarray] = None,
                  fp: Optional[np.ndarray] = None,
                  ) -> Tuple[List[RoomBlob], np.ndarray, np.ndarray]:
    """两级房间分割。

    1) 种子阶段：所有墨线 + 膨胀墙 + 门窗封堵 都当屏障，房间彼此被门洞隔开；
    2) 生长阶段：种子沿"只被墨线阻挡"的自由空间长回真实边界，面积不被膨胀影响。
    """
    if fp is None:
        fp = footprint(add_bridges(ink, bridges), grow=10)
    if seed_radius is None:
        seed_radius = int(max(7, round(0.65 * estimate_door_px(segs or []))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seed_radius + 1,) * 2)
    # 封堵要比墙略厚，否则会从窗线旁的窄缝绕过去
    barrier = add_bridges(((ink > 0) | (cv2.dilate(wall, k) > 0)).astype(np.uint8), bridges)
    barrier &= (fp > 0).astype(np.uint8)
    seed_free = (barrier == 0).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(seed_free, 4)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    seeds = np.zeros(lab.shape, np.float32)
    ids = []
    nxt = 1
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < p.min_room_px or i in border:
            continue
        seeds[lab == i] = nxt
        ids.append(nxt)
        nxt += 1

    free = ((ink == 0) & (fp > 0)).astype(np.uint8)
    rooms_lab = _propagate(seeds, free, iters=400)
    rooms_lab = _absorb_unassigned(rooms_lab, free)
    blobs, rooms_lab = build_blobs(rooms_lab, wall, ink, p, wall_vec, fp)
    return blobs, rooms_lab, fp


def build_blobs(rooms_lab: np.ndarray, wall: np.ndarray, ink: np.ndarray, p,
                wall_vec: Optional[np.ndarray] = None,
                fp: Optional[np.ndarray] = None) -> Tuple[List[RoomBlob], np.ndarray]:
    """由标签图生成房间几何：贴墙率过滤 + 家具实心块补面积 + 多边形化简。"""
    # 家具/洁具等深色实心块：不属于墙，也不该从房间面积里扣掉
    solid = solid_blocks(ink, wall_vec if wall_vec is not None else wall)

    wall_d = cv2.dilate(wall, np.ones((5, 5), np.uint8))
    # 1) 去掉"不是房间"的区域：家具/水印/阴影（边界几乎不贴墙），把像素并给邻接房间
    for rid in [int(v) for v in np.unique(rooms_lab) if v > 0]:
        m = (rooms_lab == rid).astype(np.uint8)
        if m.sum() == 0:
            continue
        if _wall_contact(m, wall_d) < 0.18:
            rooms_lab = _merge_label(rooms_lab, rid)
    keep = [int(r) for r in np.unique(rooms_lab) if r > 0]

    blobs: List[RoomBlob] = []
    for rid in keep:
        m = (rooms_lab == rid).astype(np.uint8)
        # 补回被文字/木纹/贴墙锯齿吃掉的边缘，面积才接近"净使用面积"
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        main_c = max(cnts, key=cv2.contourArea)
        # 把质心落在本房间内的实心家具块并进轮廓（面积不至于被家具吃掉）
        for ys, xs in solid:
            if cv2.pointPolygonTest(main_c, (float(xs.mean()), float(ys.mean())), False) >= 0:
                m[ys, xs] = 1
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, p.poly_eps_px, True)
        if len(approx) < 3:
            continue
        poly = orthogonalize(approx, p.ortho_snap_deg)
        if abs(cv2.contourArea(poly.astype(np.float32))) < 1:
            poly = approx.reshape(-1, 2).astype(float)
        area = float(cv2.contourArea(poly.astype(np.float32)))
        if area < p.min_room_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blobs.append(RoomBlob(m, poly, area, (x, y, w, h)))
    blobs.sort(key=lambda b: -b.area_px)
    return blobs, rooms_lab


def merge_fragments(rooms_lab: np.ndarray, blobs: List[RoomBlob],
                    has_evidence: List[bool]) -> np.ndarray:
    """把"既没有门窗、也没有房名"的封闭小块（床/沙发/餐桌围出来的坑）并回相邻房间。

    判据来自语义而不是尺寸：真正的房间一定连通别的空间（有门/窗/洞口）或写着房名。
    """
    if not blobs:
        return rooms_lab
    biggest = max(b.area_px for b in blobs)
    for i, b in enumerate(blobs):
        if has_evidence[i] or b.area_px > 0.5 * biggest:
            continue
        vals = np.unique(rooms_lab[b.mask > 0])
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        rooms_lab = _merge_label(rooms_lab, int(vals[0]))
    return rooms_lab


def _merge_label(lab: np.ndarray, rid: int) -> np.ndarray:
    """把一个标签并入与它共享边界最长的邻居。"""
    m = (lab == rid).astype(np.uint8)
    ring = (cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0) & (m == 0)
    vals = lab[ring]
    vals = vals[vals > 0]
    if vals.size == 0:
        lab[lab == rid] = 0
        return lab
    ids, cnt = np.unique(vals, return_counts=True)
    lab[lab == rid] = ids[cnt.argmax()]
    return lab


def segment_rooms(wall: np.ndarray, ink: np.ndarray, p,
                  seed_radius: Optional[int] = None,
                  segs: Optional[List[Seg]] = None,
                  bridges: Optional[List["Opening"]] = None,
                  wall_vec: Optional[np.ndarray] = None,
                  fp: Optional[np.ndarray] = None,
                  ) -> Tuple[List[RoomBlob], np.ndarray, np.ndarray]:
    """两级房间分割。

    1) 种子阶段：所有墨线 + 膨胀墙 + 门窗封堵 都当屏障，房间彼此被门洞隔开；
    2) 生长阶段：种子沿"只被墨线阻挡"的自由空间长回真实边界，面积不被膨胀影响。
    """
    if fp is None:
        fp = footprint(add_bridges(ink, bridges), grow=10)
    if seed_radius is None:
        seed_radius = int(max(7, round(0.65 * estimate_door_px(segs or []))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seed_radius + 1,) * 2)
    # 封堵要比墙略厚，否则会从窗线旁的窄缝绕过去
    barrier = add_bridges(((ink > 0) | (cv2.dilate(wall, k) > 0)).astype(np.uint8), bridges)
    barrier &= (fp > 0).astype(np.uint8)
    seed_free = (barrier == 0).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(seed_free, 4)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    seeds = np.zeros(lab.shape, np.float32)
    ids = []
    nxt = 1
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < p.min_room_px or i in border:
            continue
        seeds[lab == i] = nxt
        ids.append(nxt)
        nxt += 1

    free = ((ink == 0) & (fp > 0)).astype(np.uint8)
    rooms_lab = _propagate(seeds, free, iters=400)
    rooms_lab = _absorb_unassigned(rooms_lab, free)
    blobs, rooms_lab = build_blobs(rooms_lab, wall, ink, p, wall_vec, fp)
    return blobs, rooms_lab, fp


def build_blobs(rooms_lab: np.ndarray, wall: np.ndarray, ink: np.ndarray, p,
                wall_vec: Optional[np.ndarray] = None,
                fp: Optional[np.ndarray] = None) -> Tuple[List[RoomBlob], np.ndarray]:
    """由标签图生成房间几何：贴墙率过滤 + 家具实心块补面积 + 多边形化简。"""
    # 家具/洁具等深色实心块：不属于墙，也不该从房间面积里扣掉
    solid = solid_blocks(ink, wall_vec if wall_vec is not None else wall)

    wall_d = cv2.dilate(wall, np.ones((5, 5), np.uint8))
    # 1) 去掉"不是房间"的区域：家具/水印/阴影（边界几乎不贴墙），把像素并给邻接房间
    for rid in [int(v) for v in np.unique(rooms_lab) if v > 0]:
        m = (rooms_lab == rid).astype(np.uint8)
        if m.sum() == 0:
            continue
        if _wall_contact(m, wall_d) < 0.18:
            rooms_lab = _merge_label(rooms_lab, rid)
    keep = [int(r) for r in np.unique(rooms_lab) if r > 0]

    blobs: List[RoomBlob] = []
    for rid in keep:
        m = (rooms_lab == rid).astype(np.uint8)
        # 补回被文字/木纹/贴墙锯齿吃掉的边缘，面积才接近"净使用面积"
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        main_c = max(cnts, key=cv2.contourArea)
        # 把质心落在本房间内的实心家具块并进轮廓（面积不至于被家具吃掉）
        for ys, xs in solid:
            if cv2.pointPolygonTest(main_c, (float(xs.mean()), float(ys.mean())), False) >= 0:
                m[ys, xs] = 1
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, p.poly_eps_px, True)
        if len(approx) < 3:
            continue
        poly = orthogonalize(approx, p.ortho_snap_deg)
        if abs(cv2.contourArea(poly.astype(np.float32))) < 1:
            poly = approx.reshape(-1, 2).astype(float)
        area = float(cv2.contourArea(poly.astype(np.float32)))
        if area < p.min_room_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blobs.append(RoomBlob(m, poly, area, (x, y, w, h)))
    blobs.sort(key=lambda b: -b.area_px)
    return blobs, rooms_lab


def merge_fragments(rooms_lab: np.ndarray, blobs: List[RoomBlob],
                    has_evidence: List[bool]) -> np.ndarray:
    """把"既没有门窗、也没有房名"的封闭小块（床/沙发/餐桌围出来的坑）并回相邻房间。

    判据来自语义而不是尺寸：真正的房间一定连通别的空间（有门/窗/洞口）或写着房名。
    """
    if not blobs:
        return rooms_lab
    biggest = max(b.area_px for b in blobs)
    for i, b in enumerate(blobs):
        if has_evidence[i] or b.area_px > 0.5 * biggest:
            continue
        vals = np.unique(rooms_lab[b.mask > 0])
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        rooms_lab = _merge_label(rooms_lab, int(vals[0]))
    return rooms_lab


def _merge_label(lab: np.ndarray, rid: int) -> np.ndarray:
    """把一个标签并入与它共享边界最长的邻居。"""
    m = (lab == rid).astype(np.uint8)
    ring = (cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0) & (m == 0)
    vals = lab[ring]
    vals = vals[vals > 0]
    if vals.size == 0:
        lab[lab == rid] = 0
        return lab
    ids, cnt = np.unique(vals, return_counts=True)
    lab[lab == rid] = ids[cnt.argmax()]
    return lab


def _merge_open_pairs(lab: np.ndarray, wall_d: np.ndarray, keep: List[int], p) -> np.ndarray:
    """相邻且之间没有墙体的房间对 -> 合并（开放式厨房、与客厅连通的阳台等）。"""
    k3 = np.ones((3, 3), np.uint8)
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed = False
        rounds += 1
        for a in list(np.unique(lab)):
            if a <= 0:
                continue
            ma = (lab == a).astype(np.uint8)
            ring = (cv2.dilate(ma, k3) > 0) & (lab != a) & (lab > 0)
            vals = lab[ring]
            if vals.size < 6:
                continue
            wall_hit = ((ring) & (wall_d > 0)).sum()
            for b, cnt in zip(*np.unique(vals, return_counts=True)):
                if b <= a or cnt < 6:
                    continue
                # 该对之间的共享边界里有多少落在墙上
                pair = ring & (lab == b)
                if pair.sum() == 0:
                    continue
                share = float(((pair) & (wall_d > 0)).sum()) / float(pair.sum())
                if share < 0.2:
                    lab[lab == b] = a
                    changed = True
    return lab
