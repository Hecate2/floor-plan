"""可调参数集中管理。所有阈值都带单位/尺度含义，方便按像素分辨率标定。"""
from dataclasses import dataclass, asdict


@dataclass
class Params:
    # --- 线稿/墙体提取 ---
    wall_gray: int = 70          # 墙体（实心深色）灰度阈值
    line_gray: int = 150         # 细线（窗、门弧、家具轮廓）灰度阈值
    wall_open_ksize: int = 3     # 形态学开运算核：去掉 1-2px 细线，同时保留低清图的 3-4px 墙
    min_wall_area: int = 60      # 墙体连通域最小面积(px)
    min_wall_len: int = 12       # 矩形分解后最小边长(px)

    # --- 墙体矩形分解 / 合并 ---
    row_cover: float = 0.85      # 矩形生长时的行覆盖率要求
    snap_tol_px: int = 4         # 端点/轴网吸附容差
    merge_gap_px: int = 6        # 共线墙段合并的最大间隙

    # --- 洞口 ---
    min_opening_px: int = 8      # 洞口最小宽度(px)
    max_opening_px: int = 170    # 洞口最大宽度(px)，超过视为房间连通口
    max_opening_mm: float = 1800

    # --- 房间 ---
    min_room_px: int = 900       # 房间最小面积(px)
    poly_eps_px: float = 2.0     # 多边形简化容差
    ortho_snap_deg: float = 12.0 # 边长与该角度内视为正交

    # --- 默认建筑参数（mm），用于补齐图纸没有的信息 ---
    wall_height_mm: float = 2800
    door_height_mm: float = 2050
    window_sill_mm: float = 900
    window_head_mm: float = 2300
    exterior_wall_mm: float = 240
    interior_wall_mm: float = 120
    door_width_mm: float = 900    # 无标注时的比例锚点兜底

    def to_dict(self):
        return asdict(self)
