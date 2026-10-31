"""最小回归：能跑通 + 关键字段齐全 + 面积/比例尺在合理范围。

    python -m tests.test_smoke        # 或 pytest
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fprec.pipeline import process_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "example.webp")


def test_example():
    res = process_image(IMG)
    doc = res.document
    assert doc["units"] == "mm"
    assert doc["scale"]["mm_per_px"] > 5
    lvl = doc["level"]
    assert len(lvl["walls"]) >= 15
    assert len(lvl["rooms"]) >= 4
    assert len(lvl["openings"]) >= 3
    total = sum(r["area_m2"] for r in lvl["rooms"])
    assert 60 < total < 160, total
    for r in lvl["rooms"]:
        assert len(r["polygon"]) >= 3
        assert r["area_m2"] > 0
        assert r["centroid"][0] >= 0
    # 墙体厚度应为正且量级合理
    assert all(w["thickness"] > 40 for w in lvl["walls"])
    json.dumps(doc)          # 可序列化
    print("OK rooms=%d walls=%d openings=%d scale=%.2f total=%.1fm2"
          % (len(lvl["rooms"]), len(lvl["walls"]), len(lvl["openings"]),
             doc["scale"]["mm_per_px"], total))


if __name__ == "__main__":
    test_example()
