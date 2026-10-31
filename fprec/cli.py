"""命令行入口：

    python -m fprec.cli example.webp -o out
    python -m fprec.cli plans/*.jpg -o out --json-only
    python -m fprec.cli example.webp -o out --hints hints.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List

import cv2

from .config import Params
from .pipeline import process_image


def _expand(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                out += sorted(glob.glob(os.path.join(p, "**", ext), recursive=True))
        else:
            out.append(p)
    return [p for p in dict.fromkeys(out) if os.path.isfile(p)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("fprec", description="户型图 -> 结构化建筑数据")
    ap.add_argument("inputs", nargs="+", help="图片或目录")
    ap.add_argument("-o", "--out", default="out", help="输出目录")
    ap.add_argument("--hints", help="房间名/面积/比例尺提示 JSON（人工或 VLM 产出）")
    ap.add_argument("--json-only", action="store_true", help="只输出 JSON")
    ap.add_argument("--no-ocr", action="store_true", help="跳过 OCR")
    ap.add_argument("--split-by-labels", action="store_true",
                    help="欠分割纠正（实验）：用读到的多个房名做种子把合并区域切开")
    ap.add_argument("--wall-gray", type=int, default=70)
    ap.add_argument("--line-gray", type=int, default=150)
    a = ap.parse_args(argv)

    files = _expand(a.inputs)
    if not files:
        print("没有匹配到输入文件", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    hints = None
    if a.hints:
        with open(a.hints, "r", encoding="utf-8") as f:
            hints = json.load(f)

    p = Params(wall_gray=a.wall_gray, line_gray=a.line_gray)
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": []}
    ok = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        t0 = time.time()
        try:
            res = process_image(f, params=p, hints=hints, use_ocr=not a.no_ocr,
                                split_by_labels=a.split_by_labels)
        except Exception as e:  # 单张失败不影响批处理
            print(f"[FAIL] {f}: {e}", file=sys.stderr)
            manifest["items"].append({"file": f, "status": "failed", "error": str(e)})
            continue
        doc = {k: v for k, v in res.document.items() if not k.startswith("_")}
        jp = os.path.join(a.out, stem + ".json")
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        if not a.json_only:
            cv2.imwrite(os.path.join(a.out, stem + "_overlay.png"), res.overlay)
            with open(os.path.join(a.out, stem + ".svg"), "w", encoding="utf-8") as fh:
                fh.write(res.svg)
        d = doc["diagnostics"]
        levels = doc.get("levels") or [doc["level"]]
        print(f"[OK] {os.path.basename(f)}  plans={len(levels)} "
              f"rooms={d['stats']['rooms']} walls={d['stats']['walls']} "
              f"openings={d['stats']['openings']} {res.runtime_s:.1f}s"
              + (f"  ⚠ {len(d['warnings'])}" if d["warnings"] else ""))
        for lv in levels:
            sc = lv["scale"]
            print(f"     {lv['id']} {lv['name']}: rooms={len(lv['rooms'])} "
                  f"scale={sc['mm_per_px']}mm/px({sc['method']}, conf={sc['confidence']})")
        for w in d["warnings"]:
            print("     -", w)
        manifest["items"].append({"file": f, "status": "ok", "json": jp,
                                  "plans": len(levels),
                                  "rooms": d["stats"]["rooms"],
                                  "openings": d["stats"]["openings"],
                                  "scales": [lv["scale"] for lv in levels],
                                  "warnings": d["warnings"],
                                  "runtime_s": round(res.runtime_s, 2)})
        ok += 1
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\n完成 {ok}/{len(files)}，输出目录：{os.path.abspath(a.out)}")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
