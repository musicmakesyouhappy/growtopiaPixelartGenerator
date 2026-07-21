"""
缓存道具颜色数据
只针对"实心"道具,提前算好每个道具贴图的平均颜色 + 方差。

读取: output/solid_items.json
写入: output/item_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

from gtCommon import compute_color_stats, sprite_path

BASE_DIR = Path(__file__).resolve().parent
SPRITES_DIR = BASE_DIR / "sprites"
SOLID_FILE = BASE_DIR / "output" / "solid_items.json"
OUTPUT_FILE = BASE_DIR / "output" / "item_stats.json"


def load_solid_items() -> list[str]:
    with open(SOLID_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_all_stats(item_names: list[str]) -> dict:
    stats: dict[str, dict] = {}
    skipped_missing: list[str] = []  # 硬盘上找不到贴图文件的道具
    skipped_empty: list[str] = []  # 贴图存在,但整张全透明,没法算颜色
    failed: list[tuple[str, str]] = []  # 打开/读取贴图的时候直接报错的

    total = len(item_names)
    print(f"Processing {total} solid items...")

    for i, name in enumerate(item_names, 1):
        path = sprite_path(SPRITES_DIR, name)

        if not path.exists():
            skipped_missing.append(name)
            continue

        try:
            result = compute_color_stats(path)
        except Exception as e:  # 这里故意抓所有异常,免得某张贴图崩了就把整个脚本带崩
            failed.append((name, str(e)))
            continue

        if result is None:
            skipped_empty.append(name)
            continue

        stats[name] = result

        if i % 500 == 0:
            print(f"  {i}/{total}...")

    print(f"\nDone! {len(stats)} items processed successfully.")
    if skipped_missing:
        print(f"  {len(skipped_missing)} sprites not found on disk.")
    if skipped_empty:
        print(f"  {len(skipped_empty)} sprites had no visible (opaque) pixels.")
    if failed:
        print(f"  {len(failed)} sprites failed to process:")
        for name, err in failed[:10]:
            print(f"    - {name}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    return stats


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    solid_items = load_solid_items()
    stats = compute_all_stats(solid_items)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
