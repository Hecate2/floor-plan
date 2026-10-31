"""OCR 层：把图纸上的数字/文字读成带坐标的 token。

- 数字 -> 比例尺标定（尺寸链）与面积交叉校验
- 文字 -> 房间名 -> 房间类别（卧室/卫生间/厨房…）
后端可插拔：RapidOCR（中文、离线、自带方向检测）> tesseract（英文数字够用）。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Token:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def _clean(text: str) -> str:
    """去掉标注里的箭头/装饰符号；多个尺寸数字被连成一个 token 时丢弃（位置已不可靠）。"""
    t = text.strip().replace(" ", "").replace("㎡", "").replace("²", "").replace("°", "")
    groups = re.findall(r"\d+(?:[.,]\d+)?", t)
    if len(groups) > 1:
        return ""
    t = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff.,]", "", t)
    return re.sub(r"[mM]+$", "", t)


# ---------------------------------------------------------------- RapidOCR

_rapid = None


def _get_rapid():
    global _rapid
    if _rapid is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid = RapidOCR()
        except Exception:
            _rapid = False
    return _rapid


def ocr_rapid(gray: np.ndarray) -> List[Token]:
    eng = _get_rapid()
    if not eng:
        return []
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    # 大图先缩到 1800px 再识别（字更大，反而更稳，也快得多）
    f = max(1.0, max(bgr.shape[:2]) / 1800.0)
    if f > 1.0:
        bgr = cv2.resize(bgr, (int(bgr.shape[1] / f), int(bgr.shape[0] / f)),
                         interpolation=cv2.INTER_AREA)
    res, _ = eng(bgr)
    toks: List[Token] = []
    for box, text, conf in (res or []):
        pts = np.array(box, dtype=float)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        t = _clean(text)
        if not t:
            continue
        toks.append(Token(t, float(conf) * 100, int(round(x0 * f)), int(round(y0 * f)),
                          int(round((x1 - x0) * f)), int(round((y1 - y0) * f))))
    return toks


# ---------------------------------------------------------------- tesseract

def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_langs() -> List[str]:
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
        return [l.strip() for l in out.splitlines()[1:] if l.strip()]
    except Exception:
        return []


def ocr_tesseract(gray: np.ndarray, scale: int = 2, psm: int = 11,
                  lang: Optional[str] = None) -> List[Token]:
    if not tesseract_available():
        return []
    if lang is None:
        lang = "chi_sim+eng" if "chi_sim" in tesseract_langs() else "eng"
    tmp = tempfile.mkdtemp(prefix="fprec_ocr_")
    path = os.path.join(tmp, "img.png")
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, big)
    subprocess.run(["tesseract", path, path + ".out", "--psm", str(psm), "-l", lang, "tsv"],
                   capture_output=True, text=True)
    toks: List[Token] = []
    tsv = path + ".out.tsv"
    if os.path.exists(tsv):
        with open(tsv, "r", encoding="utf-8", errors="ignore") as f:
            for line in f.read().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) < 12 or not parts[11].strip():
                    continue
                try:
                    conf = float(parts[10])
                    x, y, w, h = (int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]))
                except ValueError:
                    continue
                if conf <= 30:
                    continue
                toks.append(Token(_clean(parts[11]), conf,
                                  *[int(round(v / scale)) for v in (x, y, w, h)]))
    shutil.rmtree(tmp, ignore_errors=True)
    return toks


# ---------------------------------------------------------------- 统一入口

def ocr(gray: np.ndarray, backend: str = "auto") -> List[Token]:
    if backend in ("auto", "rapidocr"):
        toks = ocr_rapid(gray)
        if toks:
            return toks
        if backend == "rapidocr":
            return []
    return ocr_tesseract(gray)


def _merge_tokens(a: List[Token], b: List[Token]) -> List[Token]:
    """合并两个 OCR 后端的结果；同文本且中心基本重合时保留高置信度。"""
    out = list(a)
    for t in b:
        duplicate = False
        for i, old in enumerate(out):
            if old.text != t.text:
                continue
            tol = max(6.0, 0.5 * min(old.w + old.h, t.w + t.h))
            if abs(old.cx - t.cx) <= tol and abs(old.cy - t.cy) <= tol:
                if t.conf > old.conf:
                    out[i] = t
                duplicate = True
                break
        if not duplicate:
            out.append(t)
    return out


def ocr_with_rotation(gray: np.ndarray, backend: str = "auto"):
    """返回 (tokens, 竖排补充 tokens)。自动模式融合 RapidOCR 与 tesseract。"""
    if backend == "rapidocr":
        return ocr_rapid(gray), []

    toks = ocr_tesseract(gray)
    if backend == "auto":
        toks = _merge_tokens(ocr_rapid(gray), toks)
    rot = np.rot90(gray, k=-1)                     # 顺时针 90°
    h, w = gray.shape
    mapped: List[Token] = []
    for t in ocr_tesseract(rot):
        cx, cy = t.cx, t.cy                        # R[i, j] = gray[h-1-j, i]
        nx, ny = int(round(cy)), int(round(h - 1 - cx))
        mapped.append(Token(t.text, t.conf, nx - t.h // 2, ny - t.w // 2, t.h, t.w))
    return _merge_tokens(toks, mapped), []
