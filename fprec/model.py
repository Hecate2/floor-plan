"""输出数据模型（面向后续水力/弱电/CFD 计算）。所有几何单位为 mm。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

SCHEMA = "fprec/0.2"

# 房名 -> 语义类别 / 是否水区
CATEGORY_RULES = [
    (("主卧", "卧室", "次卧", "卧房", "bed"), "bedroom", False),
    (("客厅", "起居", "living"), "living", False),
    (("餐厅", "dining"), "dining", False),
    (("厨房", "kitchen", "厨"), "kitchen", True),
    (("卫生间", "卫浴", "浴室", "洗手间", "bath", "wc"), "bathroom", True),
    (("阳台", "露台", "balcony"), "balcony", True),
    (("书房", "study"), "study", False),
    (("储物", "衣帽", "杂物", "storage"), "storage", False),
    (("玄关", "entry", "入户"), "entry", False),
    (("走廊", "过道", "corridor", "hall"), "corridor", False),
]


def classify(label: Optional[str]) -> Tuple[str, bool]:
    if not label:
        return "unknown", False
    low = label.lower()
    for keys, cat, wet in CATEGORY_RULES:
        if any(k in low for k in keys):
            return cat, wet
    return "unknown", False


@dataclass
class Wall:
    id: str
    a: Tuple[float, float]
    b: Tuple[float, float]
    thickness: float
    kind: str = "interior"      # exterior | interior | railing
    load_bearing: Optional[bool] = None
    confidence: float = 0.8

    @property
    def length(self):
        return float(((self.b[0] - self.a[0]) ** 2 + (self.b[1] - self.a[1]) ** 2) ** 0.5)


@dataclass
class Opening:
    id: str
    type: str                   # door | window | opening
    host_wall: Optional[str]
    a: Tuple[float, float]
    b: Tuple[float, float]
    width: float
    height: float
    sill: float
    confidence: float = 0.7


@dataclass
class Room:
    id: str
    label: Optional[str]
    category: str
    wet_room: bool
    polygon: List[Tuple[float, float]]
    area_m2: float
    area_printed_m2: Optional[float]
    centroid: Tuple[float, float]
    walls: List[str] = field(default_factory=list)
    openings: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class Level:
    id: str = "L1"
    name: str = "F1"
    wall_height_mm: float = 2800
    outline: List[Tuple[float, float]] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)


def build_document(image_meta: dict, levels, warnings: List[str], stats: dict) -> dict:
    """levels: [(Level, scale_dict, origin_px), ...]"""
    lv_docs = []
    for lv, scale, origin in levels:
        d = asdict(lv)
        d["scale"] = scale
        d["origin_px"] = list(origin)
        lv_docs.append(d)

    nodes, edges = [], []
    for d in lv_docs:
        lid = d["id"]
        rooms = {r["id"]: r for r in d["rooms"]}
        for r in d["rooms"]:
            nodes.append({"id": f"{lid}:{r['id']}", "level": lid, "label": r["label"],
                          "category": r["category"], "area_m2": round(r["area_m2"], 2),
                          "wet_room": r["wet_room"]})
        seen = set()
        for r in d["rooms"]:
            for nb in r["neighbors"]:
                key = tuple(sorted((r["id"], nb)))
                if key in seen or nb not in rooms:
                    continue
                seen.add(key)
                shared = [o for o in d["openings"]
                          if o["id"] in set(r["openings"]) & set(rooms[nb]["openings"])]
                edges.append({"a": f"{lid}:{key[0]}", "b": f"{lid}:{key[1]}",
                              "level": lid,
                              "via": [o["id"] for o in shared],
                              "kind": ("door" if any(o["type"] == "door" for o in shared)
                                       else "opening" if shared else "wall")})

    doc = {
        "schema": SCHEMA,
        "units": "mm",
        "coordinate_system": {"origin": "plan_bbox_min_corner", "x": "right", "y": "down"},
        "image": image_meta,
        "levels": lv_docs,
        "graph": {"nodes": nodes, "edges": edges},
        "diagnostics": {"warnings": warnings, "stats": stats},
    }
    if len(lv_docs) == 1:            # 单层时保留扁平字段，方便直接取用
        doc["level"] = lv_docs[0]
        doc["scale"] = lv_docs[0]["scale"]
    return doc
