"""fprec —— 户型图识别：图像 -> 带真实尺寸（mm）的结构化建筑数据。

对外主要入口：
    from fprec.pipeline import process_image
    res = process_image("plan.jpg")
    res.document   # dict，见 fprec.model
"""

__version__ = "0.2.0"
