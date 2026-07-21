"""
这是Growtopia像素画工具的公共小工具库。

ComputeItemStats.py 和 PixelMatcher.py 这两个脚本都会用到这里的东西,
放一起就不用两边各写一份一样的代码了。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Windows文件名不能用(或者用了很奇怪)的字符
_INVALID_CHARS = ':/\\?*<>|"'
_TRANSLATION = str.maketrans({c: "_" for c in _INVALID_CHARS})


def clean_filename(name: str) -> str:
    """把文件名里那些Windows不让用的字符,统统换成下划线。"""
    return name.translate(_TRANSLATION)


def sprite_path(sprites_dir: Path, item_name: str) -> Path:
    """根据道具的名字,拼出它贴图文件在硬盘上的路径。"""
    return sprites_dir / f"{clean_filename(item_name)}.png"


def compute_color_stats(image_path: Path, alpha_threshold: int = 64) -> Optional[dict]:
    """
    算一张贴图里"看得见"的像素的平均颜色,以及颜色的离散程度(方差)。

    alpha_threshold: 透明度超过这个值才算"看得见"的像素。
    如果整张图都是透明的,就返回 None。
    要是图片本身打不开或者读取失败,异常会直接往外抛,让上层去处理。
    """
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.int16)

    rgb = arr[..., :3].reshape(-1, 3)
    alpha = arr[..., 3].reshape(-1)
    visible = rgb[alpha > alpha_threshold]

    if visible.size == 0:
        return None

    avg = visible.mean(axis=0)
    variance = float(np.mean(np.sum((visible - avg) ** 2, axis=1)))

    return {
        "avg": [int(round(c)) for c in avg],
        "variance": round(variance, 2),
    }
