"""门窗分类、整面墙窗与边界输入的回归测试（无第三方测试依赖）。"""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fprec.config import Params
from fprec.geometry import Seg
from fprec.pipeline import _analyze, detect_plans, process_image
from fprec.scale import printed_area_tokens
from fprec.structure import (Opening, candidate_gaps, classify_openings,
                             full_wall_window_gaps)
from fprec.text import Token, _clean
from tests.make_synthetic import draw


def _footprint(shape, rect=(10, 10, 190, 190)):
    fp = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = rect
    fp[y0:y1, x0:x1] = 1
    return fp


def _classify(opening, ink, fp, mm_per_px=25.0):
    return classify_openings([opening], fp, ink, Params(), mm_per_px=mm_per_px)


class OpeningClassificationTest(unittest.TestCase):
    def test_wide_exterior_window_is_not_opening(self):
        ink = np.zeros((200, 200), np.uint8)
        for x in (21, 24, 27):
            cv2.line(ink, (x, 50), (x, 130), 1, 1)
        o = Opening("opening", "v", 24, 50, 130, 10)
        got = _classify(o, ink, _footprint(ink.shape))
        self.assertEqual([x.type for x in got], ["window"])

    def test_dense_door_leaf_is_not_window(self):
        ink = np.zeros((200, 200), np.uint8)
        cv2.line(ink, (80, 40), (80, 100), 1, 2)
        o = Opening("opening", "v", 80, 40, 100, 8)
        got = _classify(o, ink, _footprint(ink.shape))
        self.assertEqual([x.type for x in got], ["door"])

    def test_separated_internal_parallel_lines_are_window(self):
        ink = np.zeros((200, 200), np.uint8)
        for x in (74, 80, 86):
            cv2.line(ink, (x, 40), (x, 100), 1, 1)
        o = Opening("opening", "v", 80, 40, 100, 8)
        got = _classify(o, ink, _footprint(ink.shape))
        self.assertEqual([x.type for x in got], ["window"])

    def test_solid_gap_is_dropped(self):
        ink = np.zeros((200, 200), np.uint8)
        cv2.rectangle(ink, (77, 40), (83, 100), 1, -1)
        o = Opening("opening", "v", 80, 40, 100, 8)
        self.assertEqual(_classify(o, ink, _footprint(ink.shape)), [])

    def test_t_junction_splits_adjacent_gaps(self):
        p = Params()
        h1 = Seg("h", 50, 0, 50, 8, 0, 50)
        h2 = Seg("h", 50, 100, 150, 8, 100, 150)
        junction = Seg("v", 75, 20, 50, 8, 20, 50)
        got = candidate_gaps([h1, h2, junction], p)
        self.assertEqual([(o.s0, o.s1) for o in got], [(50, 71), (79, 100)])

    def test_full_wall_window_fallback(self):
        p = Params()
        ink = np.zeros((160, 160), np.uint8)
        wall = np.zeros_like(ink)
        for y in (28, 32, 36):
            cv2.line(ink, (40, y), (120, y), 1, 1)
        fp = _footprint(ink.shape, (15, 20, 145, 145))
        segs = [Seg("v", 20, 20, 140, 10, 20, 140),
                Seg("v", 140, 20, 140, 10, 20, 140)]
        found = full_wall_window_gaps(ink, wall, segs, fp, [], p)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].orient, "h")
        self.assertGreaterEqual(found[0].width, 70)
        got = _classify(found[0], ink, fp)
        self.assertEqual([x.type for x in got], ["window"])

    def test_full_wall_fallback_rejects_solid_wall_edges(self):
        p = Params()
        ink = np.zeros((160, 160), np.uint8)
        wall = np.zeros_like(ink)
        cv2.rectangle(ink, (40, 28), (120, 38), 1, -1)
        cv2.rectangle(wall, (40, 28), (120, 38), 1, -1)
        fp = _footprint(ink.shape, (15, 20, 145, 145))
        segs = [Seg("v", 20, 20, 140, 10, 20, 140),
                Seg("v", 140, 20, 140, 10, 20, 140)]
        self.assertEqual(full_wall_window_gaps(ink, wall, segs, fp, [], p), [])


class PipelineRegressionTest(unittest.TestCase):
    def test_hints_area_is_preserved_and_hints_not_mutated(self):
        img, _ = draw()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(path, img)
            hints = {"scale_mm_per_px": 20.0,
                     "rooms": [{"label": "卧室A", "area_m2": 10.0, "at": [250, 200]}]}
            res = process_image(path, hints=hints, use_ocr=False)
            rooms = res.document["level"]["rooms"]
            hit = next(r for r in rooms if r["label"] == "卧室A")
            self.assertEqual(hit["area_printed_m2"], 10.0)
            self.assertNotIn("_s", hints)
            for opening in res.document["level"]["openings"]:
                if opening["type"] == "window":
                    self.assertEqual(opening["height"], 1400)
                    self.assertEqual(opening["sill"], 900)
                else:
                    self.assertEqual(opening["height"], 2050)
                    self.assertEqual(opening["sill"], 0)
        finally:
            os.unlink(path)

    def test_blank_image_returns_empty_level_instead_of_crashing(self):
        blank = np.full((120, 160, 3), 255, np.uint8)
        result = _analyze(blank, Params(), None, use_ocr=False, split_by_labels=False)
        self.assertEqual(result.level.rooms, [])
        self.assertEqual(result.level.openings, [])
        self.assertEqual(result.level.outline, [])
        self.assertTrue(any("未提取到墙体" in w for w in result.warnings))

    def test_ocr_decimal_comma_is_preserved(self):
        self.assertEqual(_clean("11,2㎡"), "11,2")
        self.assertEqual(_clean("12.5m²"), "12.5")

    def test_integer_tokens_are_not_room_areas(self):
        toks = [Token("11.2", 95, 0, 0, 20, 10),
                Token("56", 95, 30, 0, 10, 10),
                Token("4", 95, 50, 0, 8, 10)]
        self.assertEqual([v for v, _ in printed_area_tokens(toks)], [11.2])

    def test_detect_plans_merges_fragments_of_same_floor(self):
        gray = np.full((700, 900), 255, np.uint8)
        cv2.rectangle(gray, (50, 100), (350, 600), 0, 8)
        cv2.rectangle(gray, (474, 100), (774, 600), 0, 8)
        boxes = detect_plans(gray, Params(), min_side=100)
        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][0], 1)
        self.assertGreaterEqual(boxes[0][2], 823)


if __name__ == "__main__":
    unittest.main()
