"""可视化：叠加图（PNG）与矢量导出（SVG）。"""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


def _font(size: int):
    from PIL import ImageFont
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_overlay(bgr, blobs, segs, openings, mm_per_px, level, warnings) -> np.ndarray:
    """在原图上叠加：房间着色/编号/面积、墙体、门窗洞口、比例尺。"""
    from PIL import Image, ImageDraw

    vis = bgr.copy()
    palette = [(91, 155, 213), (112, 173, 71), (255, 192, 0), (237, 125, 49),
               (165, 105, 189), (0, 176, 176), (255, 120, 120), (140, 140, 140),
               (60, 120, 200), (200, 160, 60)]
    for i, b in enumerate(blobs):
        c = np.array(palette[i % len(palette)], np.uint8)
        m = b.mask > 0
        vis[m] = (vis[m] * 0.45 + c * 0.55).astype(np.uint8)
        cv2.polylines(vis, [b.polygon.astype(np.int32)], True, (30, 30, 30), 1, cv2.LINE_AA)
    for sgm in segs:
        a, bb = sgm.endpoints()
        cv2.line(vis, (int(a[0]), int(a[1])), (int(bb[0]), int(bb[1])), (180, 20, 20), 1, cv2.LINE_AA)
    for o in openings:
        x0, y0, x1, y1 = o.rect()
        color = (0, 200, 255) if o.type == "window" else (255, 0, 200) if o.type == "door" else (0, 140, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)

    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)
    f = _font(15)
    fs = _font(13)
    for i, b in enumerate(blobs):
        room = level["rooms"][i] if i < len(level["rooms"]) else None
        if not room:
            continue
        txt = room.get("label") or room["id"]
        dr.text((b.bbox[0] + 4, b.bbox[1] + 3), txt, font=f, fill=(0, 0, 0))
        dr.text((b.bbox[0] + 4, b.bbox[1] + 21), f"{room['area_m2']:.1f} m2", font=fs, fill=(60, 60, 60))
    # 比例尺
    bar_mm = 1000.0
    bar_px = int(bar_mm / mm_per_px)
    x0, y0 = 30, bgr.shape[0] - 30
    dr.line([(x0, y0), (x0 + bar_px, y0)], fill=(0, 0, 0), width=3)
    dr.text((x0, y0 - 18), f"1 m   ({mm_per_px:.1f} mm/px)", font=fs, fill=(0, 0, 0))
    if warnings:
        dr.text((30, bgr.shape[0] - 60), warnings[0][:78], font=fs, fill=(180, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def to_svg(doc: dict) -> str:
    """把模型导成 SVG（mm 坐标，按 1/10 缩放显示，便于查看）。"""
    levels = doc.get("levels") or [doc["level"]]
    pts = []
    for lv in levels:
        pts += [p for w in lv["walls"] for p in (w["a"], w["b"])]
        pts += [p for r in lv["rooms"] for p in r["polygon"]]
    if not pts:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, miny = min(xs) - 200, min(ys) - 200
    w = max(xs) - minx + 200
    h = max(ys) - miny + 200
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{minx:.0f} {miny:.0f} {w:.0f} {h:.0f}' "
           f"width='{w/10:.0f}' height='{h/10:.0f}'>", "<g stroke-linecap='square'>"]
    for lv in levels:
        out.append(f"<g id='{lv['id']}'>")
        for r in lv["rooms"]:
            d = " ".join(f"{x:.0f},{y:.0f}" for x, y in r["polygon"])
            out.append(f"<polygon points='{d}' fill='#f2f2f2' stroke='#999' stroke-width='8'/>")
        for wl in lv["walls"]:
            (ax, ay), (bx, by) = wl["a"], wl["b"]
            kind = "#111" if wl["kind"] == "exterior" else "#555"
            out.append(f"<line x1='{ax:.0f}' y1='{ay:.0f}' x2='{bx:.0f}' y2='{by:.0f}' "
                       f"stroke='{kind}' stroke-width='{max(wl['thickness'], 60):.0f}'/>")
        for o in lv["openings"]:
            (ax, ay), (bx, by) = o["a"], o["b"]
            color = "#00a0c0" if o["type"] == "window" else "#c00040"
            out.append(f"<line x1='{ax:.0f}' y1='{ay:.0f}' x2='{bx:.0f}' y2='{by:.0f}' "
                       f"stroke='{color}' stroke-width='90'/>")
        out.append("</g>")
    out.append("</g></svg>")
    return "\n".join(out)
