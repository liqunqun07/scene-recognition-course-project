"""
合并图片文件夹：将 B/default/类别X 中的图片合并到对应的 A/类别X 中
- 只合并 A 中已存在的类别
- B 中有但 A 没有的类别会被跳过
- 支持常见图片格式：jpg, jpeg, png, gif, bmp, webp, tiff, svg
"""

import os
import shutil
import argparse
from pathlib import Path


# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg'}


def is_image(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def merge_images(folder_a: str, folder_b: str, dry_run: bool = False, rename_conflict: bool = True):
    """
    将 B/default/类别X 的图片合并到 A/类别X

    Args:
        folder_a:        A 文件夹路径
        folder_b:        B 文件夹路径
        dry_run:         True 时只打印操作，不实际复制
        rename_conflict: True 时遇到同名文件自动重命名，False 时跳过
    """
    path_a = Path(folder_a)
    path_b = Path(folder_b)

    if not path_a.exists():
        print(f"[错误] 找不到文件夹 A: {path_a}")
        return
    if not path_b.exists():
        print(f"[错误] 找不到文件夹 B: {path_b}")
        return

    # 获取 A 中的类别（子文件夹名）
    categories_a = {p.name for p in path_a.iterdir() if p.is_dir()}
    # 获取 B 中的类别：结构为 B/类别/default，所以取 B 下有 default 子目录的文件夹
    categories_b = {p.name for p in path_b.iterdir() if p.is_dir() and (p / 'default').is_dir()}

    matched   = categories_a & categories_b          # A 和 B 都有
    only_in_b = categories_b - categories_a          # 只在 B 有（跳过）

    print(f"\n📁 A 的类别数：{len(categories_a)}")
    print(f"📁 B/default 的类别数：{len(categories_b)}")
    print(f"✅ 匹配到的类别（将合并）：{len(matched)} 个 → {sorted(matched)}")
    print(f"⏭️  仅在 B 中存在（跳过）：{len(only_in_b)} 个 → {sorted(only_in_b)}")
    if dry_run:
        print("\n🔍 [演习模式] 只显示操作，不实际复制文件\n")

    total_copied = 0
    total_skipped = 0
    total_renamed = 0

    for category in sorted(matched):
        src_dir = path_b / category / 'default'
        dst_dir = path_a / category

        images = [f for f in src_dir.iterdir() if f.is_file() and is_image(f)]
        if not images:
            print(f"\n[{category}] 无图片，跳过")
            continue

        print(f"\n[{category}] 发现 {len(images)} 张图片 → 目标: {dst_dir}")

        for img in images:
            dst_file = dst_dir / img.name

            if dst_file.exists():
                if rename_conflict:
                    # 自动重命名：filename_1.jpg, filename_2.jpg ...
                    stem = img.stem
                    suffix = img.suffix
                    counter = 1
                    while dst_file.exists():
                        dst_file = dst_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    action = f"重命名复制 → {dst_file.name}"
                    total_renamed += 1
                else:
                    print(f"  ⏭️  跳过（已存在）: {img.name}")
                    total_skipped += 1
                    continue
            else:
                action = f"复制 → {dst_file.name}"

            print(f"  {'[演习]' if dry_run else ''}  {img.name}  {action}")
            if not dry_run:
                shutil.copy2(img, dst_file)
            total_copied += 1

    print("\n" + "="*50)
    print(f"✅ {'将复制' if dry_run else '已复制'}：{total_copied} 张图片")
    if rename_conflict:
        print(f"🔄 {'将重命名' if dry_run else '已重命名'}：{total_renamed} 张（同名冲突）")
    else:
        print(f"⏭️  跳过（同名冲突）：{total_skipped} 张")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='将 B/default/类别X 中的图片合并到 A/类别X（只合并 A 中已有的类别）'
    )
    parser.add_argument('folder_a', help='A 文件夹路径，例如：./A')
    parser.add_argument('folder_b', help='B 文件夹路径，例如：./B')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='演习模式：只打印操作，不实际复制文件'
    )
    parser.add_argument(
        '--skip-conflict', action='store_true',
        help='遇到同名文件时跳过（默认是自动重命名）'
    )

    args = parser.parse_args()
    merge_images(
        folder_a=args.folder_a,
        folder_b=args.folder_b,
        dry_run=args.dry_run,
        rename_conflict=not args.skip_conflict
    )


def delete_renamed_files(folder_a: str, dry_run: bool = False):
    """
    删除 A 文件夹各子类别中后缀为 _1.jpg、_2.jpg（以此类推）的文件
    """
    import re
    path_a = Path(folder_a)
    pattern = re.compile(r'.+_\d+\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif|svg)$', re.IGNORECASE)

    total_deleted = 0
    for category_dir in sorted(path_a.iterdir()):
        if not category_dir.is_dir():
            continue
        targets = [f for f in category_dir.iterdir() if f.is_file() and pattern.match(f.name)]
        if not targets:
            continue
        print(f"\n[{category_dir.name}] 找到 {len(targets)} 个文件")
        for f in targets:
            print(f"  {'[演习]' if dry_run else ''}  删除: {f.name}")
            if not dry_run:
                f.unlink()
            total_deleted += 1

    print("\n" + "="*50)
    print(f"🗑️  {'将删除' if dry_run else '已删除'}：{total_deleted} 个文件")
    print("="*50)

def count_images(folder_a: str):
    """
    统计 A 文件夹下每个子文件夹中的图片数量，降序输出
    """
    path_a = Path(folder_a)
    stats = {}
    for category_dir in path_a.iterdir():
        if not category_dir.is_dir():
            continue
        count = sum(1 for f in category_dir.iterdir() if f.is_file() and is_image(f))
        stats[category_dir.name] = count

    total = sum(stats.values())
    print(f"\n{'类别':<30} {'图片数':>6}")
    print("-" * 38)
    for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<30} {count:>6}")
    print("-" * 38)
    print(f"{'总计':<30} {total:>6}")

import random

def balance_and_split(folder_a: str, max_count: int = 150,
                       train_ratio: float = 0.6, test_ratio: float = 0.2, val_ratio: float = 0.2,
                       dry_run: bool = False):
    """
    1. 每个子文件夹超过 max_count 张则随机删除多余的
    2. 按 train/test/val 比例划分，在 folder_a 同级生成 train/test/val 文件夹
    """
    path_a = Path(folder_a)
    base = path_a.parent

    for split in ['train', 'test', 'val']:
        (base / split).mkdir(exist_ok=True)

    total_train = total_test = total_val = total_deleted = 0

    for category_dir in sorted(path_a.iterdir()):
        if not category_dir.is_dir():
            continue

        images = [f for f in category_dir.iterdir() if f.is_file() and is_image(f)]
        random.shuffle(images)

        # 超出则随机舍弃
        if len(images) > max_count:
            to_delete = images[max_count:]
            images = images[:max_count]
            print(f"[{category_dir.name}] 共 {len(images) + len(to_delete)} 张，删除 {len(to_delete)} 张")
            for f in to_delete:
                if not dry_run:
                    f.unlink()
                total_deleted += len(to_delete)

        # 按比例切分
        n = len(images)
        n_train = int(n * train_ratio)
        n_test  = int(n * test_ratio)
        # val 取剩余，避免浮点误差丢图
        splits = {
            'train': images[:n_train],
            'test':  images[n_train:n_train + n_test],
            'val':   images[n_train + n_test:]
        }

        for split, files in splits.items():
            dst_dir = base / split / category_dir.name
            dst_dir.mkdir(exist_ok=True)
            for f in files:
                dst = dst_dir / f.name
                if not dry_run:
                    shutil.copy2(f, dst)

        print(f"[{category_dir.name}] {n} 张 → train:{len(splits['train'])}  test:{len(splits['test'])}  val:{len(splits['val'])}")
        total_train += len(splits['train'])
        total_test  += len(splits['test'])
        total_val   += len(splits['val'])

    print("\n" + "=" * 50)
    print(f"🗑️  删除多余：{total_deleted} 张")
    print(f"📂 train: {total_train}  test: {total_test}  val: {total_val}  共: {total_train+total_test+total_val}")
    print(f"📁 输出目录：{base}/train  /test  /val")
    print("=" * 50)
def clean_invalid_images(folder: str, dry_run: bool = False):
    """
    扫描文件夹下所有图片，删除无法被 PIL 打开的损坏文件
    """
    from PIL import Image, UnidentifiedImageError
    path = Path(folder)
    total_checked = 0
    total_deleted = 0

    for img_path in sorted(path.rglob('*')):
        if not img_path.is_file() or not is_image(img_path):
            continue
        total_checked += 1
        try:
            with Image.open(img_path) as img:
                img.verify()  # 只校验不加载，速度快
        except (UnidentifiedImageError, Exception):
            print(f"  🗑️  损坏: {img_path}")
            if not dry_run:
                img_path.unlink()
            total_deleted += 1

    print("\n" + "=" * 50)
    print(f"✅ 检查：{total_checked} 张  🗑️  {'将删除' if dry_run else '已删除'}：{total_deleted} 张")
    print("=" * 50)

if __name__ == '__main__':
    main()
