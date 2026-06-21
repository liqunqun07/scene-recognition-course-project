#!/usr/bin/env python
"""
独立的数据预处理脚本
用于在运行实验前预先处理图像数据
将预处理逻辑从主程序中分离出来
"""
import os
import sys
import argparse
from config_loader_improved import load_config
from image_preprocessor import ImagePreprocessor


def preprocess_dataset(config_path='config.yaml', dataset_name=None, force=False):
    """
    预处理数据集
    
    参数:
        config_path: 配置文件路径
        dataset_name: 数据集名称（如果为None则使用配置文件中的默认值）
        force: 是否强制重新处理（即使输出目录已存在）
    
    返回:
        stats: 处理统计信息
    """
    # 加载配置
    print("=" * 80)
    print("数据预处理工具")
    print("=" * 80)
    
    config = load_config(config_path)
    
    # 如果指定了数据集名称，则切换到该数据集
    if dataset_name:
        config.set_dataset_name(dataset_name)
    
    # 获取配置信息
    preprocess_config = config.get_preprocessing_config()
    
    if not preprocess_config['enabled']:
        print("\n警告: 预处理在配置文件中被禁用")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            print("预处理已取消")
            return None
    
    # 获取数据集信息
    current_dataset = config.get_dataset_name()
    data_path = config.get_data_path()
    categories = config.get_categories()
    
    print(f"\n当前数据集: {current_dataset}")
    print(f"数据路径: {data_path}")
    print(f"类别数量: {len(categories)}")
    print(f"输出路径: {preprocess_config['output_path']}")
    
    # 检查输出目录
    output_path = preprocess_config['output_path']
    if os.path.exists(output_path) and not force:
        print(f"\n警告: 输出目录已存在: {output_path}")
        response = input("是否覆盖？(y/n): ")
        if response.lower() != 'y':
            print("预处理已取消")
            return None
    
    # 验证数据集
    is_valid, message = config.validate_dataset()
    if not is_valid:
        print(f"\n错误: {message}")
        return None
    
    print(f"\n数据集验证: {message}")
    
    # 创建预处理器
    preprocessor = ImagePreprocessor(preprocess_config)
    
    # 开始预处理
    print("\n" + "=" * 80)
    print("开始预处理...")
    print("=" * 80)
    
    try:
        stats = preprocessor.process_dataset(data_path, categories)
        
        # 保存处理信息
        info_file = os.path.join(output_path, 'preprocess_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("数据预处理信息\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"数据集: {current_dataset}\n")
            f.write(f"原始路径: {data_path}\n")
            f.write(f"输出路径: {output_path}\n")
            f.write(f"目标格式: {preprocess_config['target_format']}\n")
            f.write(f"转换为RGB: {preprocess_config['convert_to_rgb']}\n")
            f.write(f"调整大小: {preprocess_config['resize']}\n")
            f.write(f"\n总图像数: {stats['total']}\n")
            f.write(f"成功: {stats['success']}\n")
            f.write(f"失败: {stats['failed']}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n预处理信息已保存到: {info_file}")
        
        return stats
        
    except Exception as e:
        print(f"\n预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str, default=None,
                       help='数据集名称 (15scene, 67indoor, custom)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新处理（覆盖已存在的输出）')
    parser.add_argument('--list-datasets', action='store_true',
                       help='列出所有可用的数据集')
    
    args = parser.parse_args()
    
    # 列出数据集
    if args.list_datasets:
        config = load_config(args.config)
        datasets = config.get_available_datasets()
        print("\n可用的数据集:")
        for i, ds in enumerate(datasets, 1):
            try:
                config.set_dataset_name(ds)
                path = config.get_data_path()
                exists = "✓" if os.path.exists(path) else "✗"
                print(f"  {i}. {ds:15s} {exists} {path}")
            except Exception as e:
                print(f"  {i}. {ds:15s} ✗ (配置错误: {e})")
        return
    
    # 运行预处理
    stats = preprocess_dataset(args.config, args.dataset, args.force)
    
    if stats:
        print("\n" + "=" * 80)
        print("预处理完成！")
        print("=" * 80)
    else:
        print("\n预处理未完成或失败")
        sys.exit(1)

