"""
实心道具筛选器 v2
检查所有贴图的"填充率"(不透明像素占比)。
- 透明度阈值降到64(照顾玻璃类方块,不然会被误判成"不实心")
- 遇到损坏的图片文件会自动跳过
保存到: output/solid_items.json
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

# ============================================
# 配置
# ============================================
BASE_DIR = Path(__file__).resolve().parent
SPRITES_DIR = BASE_DIR / "sprites"
OUTPUT_FILE = BASE_DIR / "output" / "solid_items.json"
FILL_THRESHOLD = 80      # 填充率要达到百分之多少,才算"实心"
ALPHA_THRESHOLD = 64     # 像素的透明度超过这个值,才算"不透明"


# ============================================
# 检查全部贴图
# ============================================
def check_solid() -> None:
    if not SPRITES_DIR.exists():
        print(f"Sprites folder not found: {SPRITES_DIR}")
        return

    solid = []       # 填充率够高,算实心的道具
    not_solid = []   # 填充率不够,不算实心的道具
    corrupt = []      # 打开失败/文件本身就坏了的贴图

    files = sorted(SPRITES_DIR.glob("*.png"))
    total = len(files)
    print(f"Checking {total} sprites (alpha threshold: {ALPHA_THRESHOLD})...\n")

    for i, filepath in enumerate(files, 1):
        item_name = filepath.stem

        try:
            sprite = Image.open(filepath)
            sprite.verify()  # 先验证一下图片文件本身有没有问题
            sprite = Image.open(filepath).convert("RGBA")  # verify()之后文件指针失效了,得重新打开一次
            pixels = sprite.getdata()

            total_pixels = 0
            opaque_pixels = 0

            for r, g, b, a in pixels:
                total_pixels += 1
                if a > ALPHA_THRESHOLD:
                    opaque_pixels += 1

            fill_pct = (opaque_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            if fill_pct >= FILL_THRESHOLD:
                solid.append((item_name, fill_pct))
            else:
                not_solid.append((item_name, fill_pct))

        except Exception as e:
            # 这里故意抓所有异常 —— 贴图损坏的情况五花八门,
            # 与其一个个判断错误类型,不如统一记下来,跳过继续处理下一张
            corrupt.append((item_name, str(e)))
            continue

        if i % 500 == 0:
            print(f"  {i}/{total}...")

    # 按填充率从高到低排序,方便看
    solid.sort(key=lambda x: -x[1])
    not_solid.sort(key=lambda x: -x[1])

    # 保存结果(只存名字,填充率只是排序用,不需要写进文件里)
    solid_names = [name for name, _ in solid]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(solid_names, f, indent=2)

    # 汇总结果
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"  Total sprites:       {total}")
    print(f"  Solid (>={FILL_THRESHOLD}% fill):    {len(solid)}")
    print(f"  Not solid:           {len(not_solid)}")
    print(f"  Corrupt:             {len(corrupt)}")
    print(f"  Saved to:            {OUTPUT_FILE}")

    if corrupt:
        print("\n  Corrupt files (need redownload):")
        for name, err in corrupt[:10]:
            print(f"    - {name}")

    print("\nTOP 10 SOLID:")
    for name, pct in solid[:10]:
        print(f"  {pct:5.1f}%  {name}")


if __name__ == "__main__":
    check_solid()
