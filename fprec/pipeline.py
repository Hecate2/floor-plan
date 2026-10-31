"""端到端流水线：图像 -> 墙体/门窗/房间 -> 真实尺寸数据(JSON) + 可视化。"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Params
from .geometry import (Seg, decompose_rects, estimate_skew, imread_any,
                       rasterize_segs, rects_to_segs, wall_and_line_masks)
from .model import (Level, Opening as MOpening, Room as MRoom, Wall as MWall,
                    build_document, classify)
from .scale import (ScaleResult, printed_area_tokens, scale_from_chains,
                    scale_from_printed_areas)
from .structure import (RoomBlob, _propagate, add_bridges, build_blobs,
                        candidate_gaps, classify_openings, estimate_door_px,
                        footprint, full_wall_window_gaps, merge_fragments,
                        orthogonalize, segment_rooms)
from .text import Token, ocr_with_rotation
from .viz import render_overlay, to_svg


@dataclass
class PlanResult:
    document: dict
    overlay: np.ndarray
    svg: str
    rooms_px: List[RoomBlob]
    runtime_s: float


@dataclass
class SinglePlan:
    """一块平面的分析结果（多平面图里的一块）。"""
    level: Level
    overlay: np.ndarray
    blobs: List[RoomBlob]
    scale: dict
    warnings: List[str]
    stats: dict
    offset: Tuple[int, int] = (0, 0)


def _hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _assign_labels(blobs: List[RoomBlob], tokens: List[Token],
                   hints: Optional[dict]) -> Dict[int, dict]:
    """给房间挂名字/面积：优先人工/VLM hints，其次图纸上的中文标注。"""
    info: Dict[int, dict] = {i: {"label": None, "area": None, "labels": []}
                             for i in range(len(blobs))}
    # 1) OCR 到的中文房名：落在哪个房间多边形里就归给谁
    for t in tokens:
        txt = t.text.strip()
        if len(txt) < 2 or txt.isdigit():
            continue
        if not any("\u4e00" <= ch <= "\u9fff" for ch in txt):
            continue
        if classify(txt)[0] == "unknown":
            continue
        for i, b in enumerate(blobs):
            if cv2.pointPolygonTest(b.polygon.astype(np.float32), (t.cx, t.cy), False) >= 0:
                if txt not in info[i]["labels"]:
                    info[i]["labels"].append(txt)
                    info[i].setdefault("label_pts", []).append((txt, (t.cx, t.cy)))
                break
    # 2) hints（人工校对 / VLM 结构化输出）
    if hints:
        for h in hints.get("rooms", []):
            idx = None
            at = h.get("at") or h.get("center")
            if at:
                for i, b in enumerate(blobs):
                    if cv2.pointPolygonTest(b.polygon.astype(np.float32), (float(at[0]), float(at[1])), False) >= 0:
                        idx = i
                        break
            if idx is None and h.get("area_m2"):
                # 用标注面积找最接近的房间
                best = None
                for i, b in enumerate(blobs):
                    d = abs(b.area_px - h["area_m2"] * 1e6 / (hints.get("_s", 1.0) ** 2))
                    if best is None or d < best[0]:
                        best = (d, i)
                idx = best[1] if best else None
            if idx is not None:
                cur = info[idx]
                # 同一块区域命中多个 hint（说明欠分割）：取面积最大的那个当代表
                if h.get("label"):
                    if cur["label"] is None or (h.get("area_m2") or 0) > (cur.get("_hint_area") or 0):
                        cur["label"] = h["label"]
                        cur.setdefault("labels", []).append(h["label"])
                        cur["_hint_area"] = h.get("area_m2") or 0
                if h.get("area_m2"):
                    cur["area"] = float(h["area_m2"])
    for v in info.values():
        v.pop("_hint_area", None)
    return info


def _split_multi_label_rooms(blobs: List[RoomBlob], info: Dict[int, dict],
                             ink: np.ndarray, p: Params,
                             warnings: List[str]) -> Tuple[List[RoomBlob], Dict[int, dict]]:
    """一块区域里读出多个房名 => 欠分割。以房名文字位置为种子，在区域内重新测地分割。

    这是"OCR/大模型只给语义、几何仍由 CV 决定"的典型用法：文字落在哪个房间不重要，
    重要的是它提供了一个"这里属于卧室 C"的强先验。
    """
    out_blobs: List[RoomBlob] = []
    out_info: Dict[int, dict] = {}
    for i, b in enumerate(blobs):
        d = info.get(i, {})
        pts = d.get("label_pts") or []
        uniq: Dict[str, Tuple[float, float]] = {}
        for name, pt in pts:
            uniq.setdefault(name, pt)
        if len(uniq) < 2:
            out_info[len(out_blobs)] = d
            out_blobs.append(b)
            continue

        m = (b.mask > 0).astype(np.uint8)
        # 用欧氏生长（不挡线）：家具密集处会把测地距离拉偏，导致切分严重失衡
        free = m.copy()
        seeds = np.zeros(m.shape, np.float32)
        for k, (name, pt) in enumerate(uniq.items(), start=1):
            cv2.circle(seeds, (int(pt[0]), int(pt[1])), 7, k, -1)
        parts = _propagate(seeds, free, iters=800)
        pieces = []
        for k, name in enumerate(uniq, start=1):
            pm = ((parts == k)).astype(np.uint8)
            if pm.sum() < p.min_room_px:
                continue
            pieces.append((name, pm))
        if len(pieces) < 2:
            out_info[len(out_blobs)] = d
            out_blobs.append(b)
            continue
        warned = False
        for name, pm in pieces:
            pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, p.poly_eps_px, True)
            if len(approx) < 3:
                continue
            poly = orthogonalize(approx, p.ortho_snap_deg)
            area = abs(float(cv2.contourArea(poly.astype(np.float32))))
            x, y, w, h = cv2.boundingRect(c)
            out_info[len(out_blobs)] = {"label": name, "labels": [name],
                                        "area": None, "label_pts": [(name, uniq[name])]}
            out_blobs.append(RoomBlob(pm, poly, area, (x, y, w, h)))
            if not warned:
                warnings.append(f"{i + 1} 号区域含多个房名，已按房名种子拆分为 "
                                f"{'/'.join(n for n, _ in pieces)}")
                warned = True
    return out_blobs, out_info


def _analyze(bgr: np.ndarray, p: Params, hints: Optional[dict], use_ocr: bool,
             split_by_labels: bool, level_id: str = "L1",
             level_name: str = "F1") -> "SinglePlan":
    """单张平面（可能是整图，也可能是多平面图里裁出来的一块）。"""
    warnings: List[str] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    H, W = gray.shape

    # --- 1. 墙体矢量化 ---
    wall, lines = wall_and_line_masks(gray, p)
    ink = ((lines > 0) | (wall > 0)).astype(np.uint8)
    rects = decompose_rects(wall, p.min_wall_len)
    segs = rects_to_segs(rects, p)
    wall_vec = rasterize_segs(segs, wall.shape)
    if not segs:
        warnings.append("未提取到墙体，请检查阈值/分辨率")

    # --- 2. OCR 与比例尺（先标定，后面判门窗宽度要用到 mm） ---
    tokens: List[Token] = []
    if use_ocr:
        tokens, rot = ocr_with_rotation(gray)
        tokens = tokens + rot
    sr: Optional[ScaleResult] = scale_from_chains(tokens)
    if sr is None:
        # 兜底：按"外墙 240mm"的经验比例先跑一遍，后面再用标注面积拟合
        wall_px_sorted = sorted([s.t for s in segs], reverse=True) or [6.0]
        ext_t = float(np.median(wall_px_sorted[:max(1, len(wall_px_sorted) // 3)]))
        sr = ScaleResult(240.0 / max(ext_t, 1.0), "default_wall", 0.25, [],
                         f"无标注链，按外墙 240mm / {ext_t:.1f}px 反推（占位）")
    s = sr.mm_per_px
    if hints and hints.get("scale_mm_per_px"):
        s = float(hints["scale_mm_per_px"])
        sr = ScaleResult(s, "hint", 0.99, [s], "人工/外部给定")

    # --- 3. 轮廓 + 洞口 ---
    cands = candidate_gaps(segs, p)
    fp = footprint(add_bridges(ink, cands), grow=10)
    opening_cands = cands + full_wall_window_gaps(ink, wall, segs, fp, cands, p)
    openings = classify_openings(opening_cands, fp, ink, p, mm_per_px=s)
    room_openings = classify_openings(cands, fp, ink, p, mm_per_px=s)

    # --- 4. 房间 ---
    # 种子阶段用"全部候选缺口"封堵（比成品洞口更宽松），避免宽窗洞把两个房间连起来
    blobs, rooms_lab, fp = segment_rooms(wall, ink, p, segs=segs,
                                         bridges=cands, wall_vec=wall_vec, fp=fp)

    # 4b. 家具围出来的假房间：既没有门窗也读不到房名 -> 并回邻接房间
    name_tokens = [t for t in tokens
                   if len(t.text) >= 2 and any("\u4e00" <= ch <= "\u9fff" for ch in t.text)]
    evidence = []
    for b in blobs:
        has_name = any(cv2.pointPolygonTest(b.polygon.astype(np.float32), (t.cx, t.cy), False) >= 0
                       for t in name_tokens)
        near = cv2.dilate(b.mask, np.ones((5, 5), np.uint8))
        has_open = False
        for o in room_openings:
            x0, y0, x1, y1 = o.rect()
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            if 0 <= cy < near.shape[0] and 0 <= cx < near.shape[1] and near[cy, cx]:
                has_open = True
                break
        evidence.append(has_name or has_open)
    if sum(1 for e in evidence if not e) > 0 and len(blobs) > 2:
        rooms_lab = merge_fragments(rooms_lab, blobs, evidence)
        blobs, rooms_lab = build_blobs(rooms_lab, wall, ink, p, wall_vec, fp)
        warnings.append(f"合并了 {sum(1 for e in evidence if not e)} 个无门无名的封闭碎块"
                        f"（家具/水槽围出来的坑）")
    if len(blobs) < 2:
        warnings.append(f"仅分割出 {len(blobs)} 个房间，连通/闭合可能失败")

    # 用图上标注面积兜底/校验比例尺
    area_tokens = printed_area_tokens(tokens)
    printed_by_room = {}
    if area_tokens and blobs:
        pairs = []
        for val, t in area_tokens:
            for i, b in enumerate(blobs):
                if cv2.pointPolygonTest(b.polygon.astype(np.float32), (t.cx, t.cy), False) >= 0:
                    pairs.append((val, b.area_px))
                    printed_by_room[i] = val
                    break
        if sr.method == "default_wall":
            alt = scale_from_printed_areas(pairs)
            if alt:
                sr, s = alt, alt.mm_per_px
    # 标注面积可能替换比例尺，门宽/窗宽与类别必须使用最终比例尺。
    openings = classify_openings(opening_cands, fp, ink, p, mm_per_px=s)
    if sr.method == "default_wall":
        warnings.append("未能从图中读出比例尺，使用墙厚经验值（误差可能 >30%）")
    label_hints = dict(hints) if hints else None
    if label_hints is not None:
        label_hints["_s"] = s

    # --- 5. 房间语义 ---
    info = _assign_labels(blobs, tokens, label_hints)
    # 欠分割纠正：一块区域里读出多个房名时，用这些房名做种子重新切一刀（墙/线仍能阻挡生长）
    if split_by_labels:
        blobs, info = _split_multi_label_rooms(blobs, info, ink, p, warnings)
    # 拆分后房间索引变了，重新挂 OCR 面积；没有 OCR 面积时保留 hint 面积。
    hint_areas = {i: d.get("area") for i, d in info.items() if d.get("area") is not None}
    for i in info:
        info[i]["area"] = None
    cand: Dict[int, List[float]] = {}
    for val, t in area_tokens:
        for i, b in enumerate(blobs):
            if cv2.pointPolygonTest(b.polygon.astype(np.float32), (t.cx, t.cy), False) >= 0:
                cand.setdefault(i, []).append(val)
                break
    for i, vals in cand.items():
        # 一块区域里出现多个面积标注：通常是开放空间被合并（客厅+厨房），面积按求和核对
        info.setdefault(i, {"label": None, "area": None, "labels": []})["area"] = round(sum(vals), 2)
    for i, d in info.items():
        if d["labels"] and not d.get("label"):
            d["label"] = "+".join(d["labels"])
            if len(d["labels"]) > 1:
                warnings.append(f"{i + 1} 号区域内含多个房名 {'/'.join(d['labels'])}，"
                                "多半是欠分割（两个房间被连成一块）")
    for i, val in printed_by_room.items():
        if info.get(i, {}).get("area") is None:
            info.setdefault(i, {"label": None, "area": None})["area"] = val
    for i, val in hint_areas.items():
        if info.get(i, {}).get("area") is None:
            info[i]["area"] = val

    # --- 6. 组装毫米级模型 ---
    xs = [0]; ys = [0]
    for sgm in segs:
        for (x, y) in sgm.endpoints():
            xs.append(x); ys.append(y)
    ox, oy = min(xs), min(ys)

    def to_mm(pt):
        return (round((pt[0] - ox) * s, 1), round((pt[1] - oy) * s, 1))

    # 外墙/内墙判定：靠近建筑轮廓边界即外墙
    edge = cv2.morphologyEx(fp, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    walls_mm: List[MWall] = []
    for i, sg in enumerate(segs):
        a_px, b_px = sg.endpoints()
        mid = ((a_px[0] + b_px[0]) / 2, (a_px[1] + b_px[1]) / 2)
        near_edge = False
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                xx, yy = int(mid[0] + dx), int(mid[1] + dy)
                if 0 <= yy < H and 0 <= xx < W and edge[yy, xx] > 0:
                    near_edge = True
        walls_mm.append(MWall(
            id=f"W{i + 1}", a=to_mm(a_px), b=to_mm(b_px),
            thickness=round(sg.t * s, 0),
            kind="exterior" if near_edge else "interior",
            confidence=0.85 if sg.length > 25 else 0.6))

    def nearest_wall(pt) -> Optional[str]:
        best = None
        for w in walls_mm:
            ax, ay = w.a; bx, by = w.b
            mx, my = to_mm(pt)
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            t = max(0, min(1, ((mx - ax) * dx + (my - ay) * dy) / L2))
            px, py = ax + t * dx, ay + t * dy
            d = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
            if best is None or d < best[0]:
                best = (d, w.id)
        return best[1] if best else None

    openings_mm: List[MOpening] = []
    for i, o in enumerate(openings):
        if o.orient == "h":
            a_px, b_px = (o.s0, o.line), (o.s1, o.line)
        else:
            a_px, b_px = (o.line, o.s0), (o.line, o.s1)
        width_mm = o.width * s
        is_window = o.type == "window"
        height = (p.window_head_mm - p.window_sill_mm) if is_window else p.door_height_mm
        sill = p.window_sill_mm if is_window else 0.0
        openings_mm.append(MOpening(
            id=f"O{i + 1}", type=o.type, host_wall=nearest_wall(o.center),
            a=to_mm(a_px), b=to_mm(b_px), width=round(width_mm, 0),
            height=height, sill=sill, confidence=o.confidence))

    # 房间多边形（px -> mm），并统计相邻墙/洞口/房间
    rooms_mm: List[MRoom] = []
    for i, b in enumerate(blobs):
        poly = [to_mm(pt) for pt in b.polygon.reshape(-1, 2)]
        area_m2 = b.area_px * s * s / 1e6
        label = info[i]["label"]
        cat, wet = classify(label)
        cx, cy = [float(v) for v in b.polygon.reshape(-1, 2).mean(axis=0)]
        rooms_mm.append(MRoom(
            id=f"R{i + 1}", label=label, category=cat, wet_room=wet,
            polygon=poly, area_m2=round(area_m2, 2),
            area_printed_m2=info[i]["area"],
            centroid=to_mm((cx, cy)),
            confidence=round(min(0.9, 0.5 + 0.4 * min(1.0, b.area_px / 20000)), 2)))

    # 相邻关系：房间掩码膨胀相交 + 洞口归属
    for i, b in enumerate(blobs):
        di = cv2.dilate(b.mask, np.ones((9, 9), np.uint8))
        for j, b2 in enumerate(blobs):
            if i >= j:
                continue
            inter = int((di & cv2.dilate(b2.mask, np.ones((9, 9), np.uint8))).sum())
            if inter > 200:
                rooms_mm[i].neighbors.append(rooms_mm[j].id)
                rooms_mm[j].neighbors.append(rooms_mm[i].id)
    for o in openings_mm:
        mx = (o.a[0] + o.b[0]) / 2
        my = (o.a[1] + o.b[1]) / 2
        px, py = mx / s + ox, my / s + oy
        touching = []
        for i, b in enumerate(blobs):
            if cv2.pointPolygonTest(b.polygon.astype(np.float32), (px, py), True) > -10:
                touching.append(i)
        for i in touching:
            rooms_mm[i].openings.append(o.id)
        if len(touching) == 2:
            a, bb = rooms_mm[touching[0]], rooms_mm[touching[1]]
            if bb.id not in a.neighbors:
                a.neighbors.append(bb.id)
                bb.neighbors.append(a.id)
    for w in walls_mm:
        mx = (w.a[0] + w.b[0]) / 2
        my = (w.a[1] + w.b[1]) / 2
        for i, b in enumerate(blobs):
            if cv2.pointPolygonTest(b.polygon.astype(np.float32),
                                    (mx / s + ox, my / s + oy), True) > -14:
                rooms_mm[i].walls.append(w.id)

    # --- 7. 质量自检 ---
    for r in rooms_mm:
        if r.area_printed_m2:
            err = (r.area_m2 - r.area_printed_m2) / r.area_printed_m2
            if abs(err) > 0.15:
                warnings.append(
                    f"{r.id} 面积偏差 {err:+.0%}（计算 {r.area_m2:.1f} vs 标注 {r.area_printed_m2}），"
                    "可能是分割中被家具/墙线吃掉面积")
    if not tokens:
        warnings.append("OCR 未获得任何文字（未安装 tesseract？），比例尺与房名不可用")
    if not openings:
        warnings.append("未检测到门窗洞口")

    contours, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = []
    if contours:
        outline = [to_mm(pt) for pt in cv2.approxPolyDP(
            max(contours, key=cv2.contourArea), 3, True).reshape(-1, 2)]
    level = Level(id=level_id, name=level_name, wall_height_mm=p.wall_height_mm,
                  walls=walls_mm, openings=openings_mm, rooms=rooms_mm,
                  outline=outline)
    stats = {
        "rooms": len(rooms_mm), "walls": len(walls_mm), "openings": len(openings_mm),
        "wall_px": int(wall.sum()), "segments": len(segs), "ocr_tokens": len(tokens),
    }
    scale = {"mm_per_px": round(s, 3), "method": sr.method, "confidence": sr.confidence,
             "samples": [round(v, 2) for v in sr.samples[:12]], "detail": sr.detail}
    room_meta = [{"label": r.label, "area_m2": r.area_m2, "id": r.id} for r in rooms_mm]
    overlay = render_overlay(bgr, blobs, segs, openings, s, {"rooms": room_meta}, warnings)
    return SinglePlan(level=level, overlay=overlay, blobs=blobs, scale=scale,
                      warnings=warnings, stats=stats)


def detect_plans(gray: np.ndarray, p: Params, min_side: int = 180,
                 min_area_frac: float = 0.005) -> List[Tuple[int, int, int, int]]:
    """一张图上可能并排放着几个平面（如"1F / 2F"）。用墙体聚成块来切。

    只看墙体不看墨线：尺寸链是细线，不会被误当成一块平面。
    """
    H, W = gray.shape
    wall, _ = wall_and_line_masks(gray, p)
    if int(wall.sum()) == 0:
        return [(0, 0, W, H)]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(wall, k), 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]))
        if w < min_side or h < min_side or int(st[i, 4]) < min_area_frac * gray.size:
            continue
        m = wall[y:y + h, x:x + w]
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            continue
        pad = 60        # 要带上外围尺寸链
        boxes.append([max(0, x + xs.min() - pad), max(0, y + ys.min() - pad),
                      min(W, x + xs.max() + pad), min(H, y + ys.max() + pad)])
    if len(boxes) < 2:
        return [(0, 0, W, H)]
    boxes.sort(key=lambda b: b[0])
    merged = [boxes[0]]
    for b in boxes[1:]:                     # 合并重叠或仅有裁切缝隙的框
        if b[0] <= merged[-1][2] + 24:
            merged[-1] = [min(merged[-1][0], b[0]), min(merged[-1][1], b[1]),
                          max(merged[-1][2], b[2]), max(merged[-1][3], b[3])]
        else:
            merged.append(b)
    if len(merged) < 2:
        return [(0, 0, W, H)]
    return [tuple(int(v) for v in b) for b in merged]


def process_image(path: str, params: Optional[Params] = None,
                  hints: Optional[dict] = None, use_ocr: bool = True,
                  split_by_labels: bool = False) -> PlanResult:
    """读图 -> （必要时）切成多块平面 -> 逐块分析 -> 合成一个文档。"""
    p = params or Params()
    t0 = time.time()
    bgr = imread_any(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    boxes = detect_plans(gray, p)
    plans: List[SinglePlan] = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        sub = bgr[y0:y1, x0:x1]
        name = f"F{i + 1}" if len(boxes) > 1 else "F1"
        sp = _analyze(sub, p, hints, use_ocr, split_by_labels,
                      level_id=f"L{i + 1}", level_name=name)
        sp.offset = (x0, y0)
        plans.append(sp)

    overlay = bgr.copy()
    for sp in plans:
        x0, y0 = sp.offset
        h, w = sp.overlay.shape[:2]
        overlay[y0:y0 + h, x0:x0 + w] = sp.overlay
        if len(plans) > 1:
            cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 160), 2)
            cv2.putText(overlay, f"{sp.level.id} {sp.level.name}", (x0 + 8, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 160), 2)

    warnings: List[str] = []
    for sp in plans:
        warnings.extend(f"[{sp.level.id}] {w}" for w in sp.warnings)
    if len(plans) > 1:
        warnings.insert(0, f"检测到 {len(plans)} 块平面（可能是多楼层/多户型并排），已分别识别")

    stats = {"plans": len(plans),
             "rooms": sum(s.stats["rooms"] for s in plans),
             "walls": sum(s.stats["walls"] for s in plans),
             "openings": sum(s.stats["openings"] for s in plans),
             "thresholds": p.to_dict()}
    doc = build_document(
        {"file": os.path.basename(path), "sha256": _hash(path),
         "width_px": W, "height_px": H},
        [(sp.level, sp.scale, list(sp.offset)) for sp in plans],
        warnings, stats)
    svg = to_svg(doc)
    return PlanResult(doc, overlay, svg, [b for sp in plans for b in sp.blobs],
                      time.time() - t0)
