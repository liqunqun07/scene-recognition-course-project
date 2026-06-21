import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def simple_stats(base_path):
    """
    简单统计tourism_pic文件夹中train/val/test的图片分布
    """
    base_dir = Path(base_path)
    splits = ['train', 'val', 'test']

    # 存储统计数据
    stats = []

    print(f"\n{'=' * 60}")
    print(f"旅游场景数据集图片统计")
    print(f"{'=' * 60}")

    total_train = total_val = total_test = 0

    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"警告：{split_dir} 不存在")
            continue

        print(f"\n📁 {split.upper()} 文件夹:")
        print(f"{'-' * 40}")

        split_total = 0
        # 遍历每个类别文件夹
        for category_dir in sorted(split_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            # 统计图片数量
            image_count = sum(1 for f in category_dir.iterdir()
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])

            if image_count > 0:
                print(f"  {category_dir.name:<30} {image_count:>4}张")
                split_total += image_count

                # 记录到总统计
                stats.append({
                    '类别': category_dir.name,
                    '数据集': split,
                    '数量': image_count
                })

        print(f"{'-' * 40}")
        print(f"  {split.upper()} 总计: {split_total}张")

        # 累加总数
        if split == 'train':
            total_train = split_total
        elif split == 'val':
            total_val = split_total
        else:
            total_test = split_total

    # 打印总统计
    print(f"\n{'=' * 60}")
    print(f"数据集总体统计")
    print(f"{'=' * 60}")
    print(f"训练集 (train): {total_train}张")
    print(f"验证集 (val):   {total_val}张")
    print(f"测试集 (test):  {total_test}张")
    print(f"{'-' * 40}")
    print(f"总计:          {total_train + total_val + total_test}张")
    print(f"{'=' * 60}")

    return stats, total_train, total_val, total_test


def plot_simple_distribution(stats):
    """
    绘制简单的分布图
    """
    # 整理数据
    df = pd.DataFrame(stats)

    # 获取所有类别
    categories = df['类别'].unique()

    # 为每个类别获取train/val/test的数量
    data = []
    for cat in categories:
        cat_data = {'类别': cat}
        for split in ['train', 'val', 'test']:
            count = df[(df['类别'] == cat) & (df['数据集'] == split)]['数量'].sum()
            cat_data[split] = count
        data.append(cat_data)

    plot_df = pd.DataFrame(data)
    plot_df = plot_df.sort_values('类别')

    # 绘制柱状图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：各类别分布
    ax1 = axes[0]
    x = range(len(plot_df))
    width = 0.25

    ax1.bar([i - width for i in x], plot_df['train'], width, label='train', color='#2E86AB', alpha=0.8)
    ax1.bar(x, plot_df['val'], width, label='val', color='#A23B72', alpha=0.8)
    ax1.bar([i + width for i in x], plot_df['test'], width, label='test', color='#F18F01', alpha=0.8)

    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('图片数量', fontsize=12)
    ax1.set_title('各类别图片数量分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df['类别'], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

    # 右图：总体分布
    ax2 = axes[1]
    totals = [plot_df['train'].sum(), plot_df['val'].sum(), plot_df['test'].sum()]
    labels = ['训练集', '验证集', '测试集']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    wedges, texts, autotexts = ax2.pie(totals, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 12})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('数据集划分比例', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✅ 图表已保存为: dataset_distribution.png")


def main():
    # 设置路径
    base_path = os.environ.get("TOURISM_DATA_PATH", "./tourism_pic")
    # 统计
    stats, total_train, total_val, total_test = simple_stats(base_path)
    # 绘制图表
    plot_simple_distribution(stats)

    # 可选：保存详细统计到CSV
    df = pd.DataFrame(stats)
    df.to_csv('tourism_stats.csv', index=False, encoding='utf-8-sig')
    print("\n📊 详细统计已保存至: tourism_stats.csv")


if __name__ == "__main__":
    main()
