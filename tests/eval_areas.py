"""评估：把每张图上"计算面积"与"图上标注面积"对比，给出误差分布。

    python -m tests.eval_areas example.webp example2.jpg ...
    python -m tests.eval_areas examples/          # 目录

输出：逐图逐房间误差 + 汇总（中位绝对误差、±10% 内占比、比例尺来源）。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fprec.pipeline import process_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def eval_one(path):
    res = process_image(path, use_ocr=True)
    doc = res.document
    rows = []
    for lv in doc["levels"]:
        for r in lv["rooms"]:
            p = r.get("area_printed_m2")
            if not p:
                continue
            rows.append({
                "level": lv["id"], "id": r["id"], "label": r["label"],
                "category": r["category"],
                "calc": r["area_m2"], "printed": p,
                "err": (r["area_m2"] - p) / p,
            })
    return doc, rows


def main(argv):
    files = []
    for a in argv:
        if os.path.isdir(a):
            for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png"):
                files += sorted(glob.glob(os.path.join(a, ext)))
        else:
            files.append(a)
    if not files:
        files = sorted(f for ext in ("webp", "jpg", "jpeg", "png")
                       for f in glob.glob(os.path.join(ROOT, f"example*.{ext}")))

    all_err = []
    print(f"{'file':16} {'plans':>5} {'rooms':>5} {'scale':>7} {'method':14} "
          f"{'paired':>6} {'medAE':>6} {'<10%':>5}")
    for f in files:
        doc, rows = eval_one(f)
        lv = doc["levels"]
        errs = [abs(r["err"]) for r in rows]
        all_err += errs
        med = sorted(errs)[len(errs) // 2] if errs else 0
        within = sum(1 for e in errs if e <= 0.10) / len(errs) if errs else 0
        s = lv[0]["scale"]
        print(f"{os.path.basename(f):16} {len(lv):5d} "
              f"{sum(len(x['rooms']) for x in lv):5d} {s['mm_per_px']:7.2f} "
              f"{s['method']:14} {len(rows):6d} {med:6.1%} {within:5.0%}")
    if all_err:
        all_err.sort()
        n = len(all_err)
        print(f"\n合计 {n} 个带标注的房间：中位绝对误差 {all_err[n // 2]:.1%}，"
              f"≤10% {sum(1 for e in all_err if e <= 0.10) / n:.0%}，"
              f"≤20% {sum(1 for e in all_err if e <= 0.20) / n:.0%}，"
              f"≤30% {sum(1 for e in all_err if e <= 0.30) / n:.0%}")


if __name__ == "__main__":
    main(sys.argv[1:])
