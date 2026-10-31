"""生成一张"线稿风格"的合成户型图（含尺寸链、门弧、双线窗），用于验证算法不只会认那一张渲染图。

    python -m tests.make_synthetic            # 写 out/synthetic.png + 真值
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 900, 640
PW = 8                    # 墙厚 px
S = 20.0                  # mm/px（合成真值）

# 平面：外框 (120,80)-(760,540)，竖墙 x=390 分左右，横墙 y=330 分上下
rooms = [
    ("卧室A", (120, 80, 390, 330)),
    ("客厅", (120, 330, 520, 540)),
    ("卧室B", (390, 80, 760, 250)),
    ("卫生间", (390, 250, 560, 330)),
    ("厨房", (560, 250, 760, 540)),
]


def _font(size):
    for f in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def draw():
    img = np.full((H, W, 3), 255, np.uint8)

    def wall(x0, y0, x1, y1):
        cv2.rectangle(img, (x0, y0), (x1, y1), (10, 10, 10), -1)

    def hwall(x0, x1, y, gap=None):
        if gap is None:
            wall(x0, y - PW // 2, x1, y + PW // 2)
        else:
            g0, g1 = gap
            wall(x0, y - PW // 2, g0, y + PW // 2)
            wall(g1, y - PW // 2, x1, y + PW // 2)

    def vwall(y0, y1, x, gap=None):
        if gap is None:
            wall(x - PW // 2, y0, x + PW // 2, y1)
        else:
            g0, g1 = gap
            wall(x - PW // 2, y0, x + PW // 2, g0)
            wall(x - PW // 2, g1, x + PW // 2, y1)

    # 外墙（带窗：双细线）
    hwall(120, 390, 80, gap=(180, 330))          # 卧室A 窗
    hwall(390, 760, 80, gap=(470, 690))          # 卧室B 窗
    vwall(80, 540, 120)
    vwall(80, 540, 760, gap=(330, 470))          # 厨房窗
    hwall(120, 520, 540, gap=(360, 420))         # 入户门
    hwall(520, 760, 540)
    # 内墙（带门洞）
    vwall(80, 330, 390, gap=(250, 290))          # 卧室A-卧室B 门
    hwall(120, 390, 330, gap=(250, 290))         # 卧室A-客厅 门
    hwall(390, 760, 250, gap=(600, 640))         # 卧室B-厨房 门
    vwall(250, 330, 560)
    hwall(390, 560, 330, gap=(440, 480))         # 卫生间门

    # 窗符号：双细线
    for (x0, x1, y) in ((180, 330, 80), (470, 690, 80)):
        for dy in (-3, 3):
            cv2.line(img, (x0, y + dy), (x1, y + dy), (60, 60, 60), 1)
    for (y0, y1, x) in ((330, 470, 760),):
        for dx in (-3, 3):
            cv2.line(img, (x + dx, y0), (x + dx, y1), (60, 60, 60), 1)

    # 门弧（1/4 圆 + 门扇）
    cv2.ellipse(img, (290, 330), (40, 40), 0, 90, 180, (90, 90, 90), 1)
    cv2.line(img, (250, 330), (250, 290), (90, 90, 90), 1)
    cv2.ellipse(img, (390, 290), (40, 40), 0, 0, 90, (90, 90, 90), 1)
    cv2.ellipse(img, (600, 250), (40, 40), 0, -90, 0, (90, 90, 90), 1)
    cv2.ellipse(img, (440, 330), (40, 40), 0, -90, 0, (90, 90, 90), 1)
    cv2.ellipse(img, (420, 540), (60, 60), 0, 180, 270, (90, 90, 90), 1)

    # 房名 + 面积（真值由矩形面积算，口径：净尺寸）
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    gt = {}
    for name, (x0, y0, x1, y1) in rooms:
        iw = (x1 - x0 - PW) * S / 1000.0
        ih = (y1 - y0 - PW) * S / 1000.0
        area = round(iw * ih, 1)
        gt[name] = {"area_m2": area, "rect_mm": [round((x1 - x0 - PW) * S), round((y1 - y0 - PW) * S)]}
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        f = _font(18)
        dr.text((cx - 26, cy - 16), name, font=f, fill=(0, 0, 0))
        dr.text((cx - 26, cy + 8), f"{area}m²", font=_font(14), fill=(0, 0, 0))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 顶部尺寸链（数字放在每段中间，与真实尺寸一致）
    xs = [120, 260, 390, 560, 760]
    y = 40
    cv2.line(img, (xs[0], y), (xs[-1], y), (0, 0, 0), 1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    for i in range(len(xs) - 1):
        cx = (xs[i] + xs[i + 1]) // 2
        mm = int((xs[i + 1] - xs[i]) * S)
        cv2.line(img, (xs[i], y - 6), (xs[i], y + 6), (0, 0, 0), 1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        dr.text((cx - 16, y - 20), str(mm), font=_font(14), fill=(0, 0, 0))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.line(img, (xs[-1], y - 6), (xs[-1], y + 6), (0, 0, 0), 1)
    return img, {"scale_mm_per_px": S, "rooms": gt, "rect_px": {n: r for n, r in rooms}}


if __name__ == "__main__":
    out = os.path.join(ROOT, "out")
    os.makedirs(out, exist_ok=True)
    img, gt = draw()
    cv2.imwrite(os.path.join(out, "synthetic.png"), img)
    with open(os.path.join(out, "synthetic_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print("written:", os.path.join(out, "synthetic.png"))
