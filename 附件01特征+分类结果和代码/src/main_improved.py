#!/usr/bin/env python
"""
场景识别主程序 - 改进版
支持通过数据集名称自动选择路径
结果不会被覆盖（使用时间戳）
预处理逻辑已独立到单独脚本
"""
import numpy as np
import os
import sys
from datetime import datetime
import shutil
import json
import argparse

# 导入配置加载器
from config_loader_improved import load_config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.getcwd()  # 获取当前工作目录
print(f"当前工作目录: {project_root}")
# 切换到脚本目录，保证相对配置路径和同目录模块导入稳定。
os.chdir(SCRIPT_DIR)
# 导入其他模块
from helpers_updated import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary_fixed import build_vocabulary
from get_bags_of_words_fixed import get_bags_of_words
from get_cnn_features_fixed import get_cnn_features
from nearest_neighbor_classify import nearest_neighbor_classify
from create_results_webpage_updated import create_results_webpage
from svm_classify import svm_classify

# 尝试导入SVM分类器（如果没有则跳过）
try:
    from svm_classify import svm_classify
    HAS_SVM = True
except:
    HAS_SVM = False
    print("警告: 未找到svm_classify模块，SVM分类器将不可用")


def calculate_accuracy(predicted_categories, test_labels):
    """计算分类准确率"""
    correct = sum([1 for pred, true in zip(predicted_categories, test_labels) if pred == true])
    total = len(test_labels)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def get_feature_name(feature, cnn_model=None):
    """获取特征方法的显示名称"""
    if feature.lower() == 'tiny_image':
        return 'TinyImage'
    elif feature.lower() == 'bag_of_words':
        return 'BoW'
    elif feature.lower() == 'cnn':
        if cnn_model:
            return f'CNN_{cnn_model.upper()}'
        return 'CNN'
    else:
        return feature


def get_classifier_name(classifier):
    """获取分类器的显示名称"""
    if classifier.lower() == 'nearest_neighbor':
        return 'NN'
    elif classifier.lower() == 'support_vector_machine':
        return 'SVM'
    else:
        return classifier


def save_results_to_folder(result_folder, train_image_paths, test_image_paths, 
                          train_labels, test_labels, predicted_categories,
                          feature, classifier, cnn_model, accuracy, dataset_name):
    """保存实验结果到指定文件夹"""
    # 保存配置信息
    config_info = {
        'dataset': dataset_name,
        'feature_method': feature,
        'cnn_model': cnn_model if feature.lower() == 'cnn' else 'N/A',
        'classifier': classifier,
        'accuracy': accuracy,
        'num_train_images': len(train_image_paths),
        'num_test_images': len(test_image_paths),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_file = os.path.join(result_folder, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)
    print(f"配置信息已保存到: {config_file}")
    
    # 保存详细结果文本
    results_txt = os.path.join(result_folder, 'results.txt')
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("场景识别实验结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【实验配置】\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"特征提取方法: {feature}\n")
        if feature.lower() == 'cnn':
            f.write(f"CNN模型: {cnn_model}\n")
        f.write(f"分类器: {classifier}\n")
        f.write(f"训练图片数量: {len(train_image_paths)}\n")
        f.write(f"测试图片数量: {len(test_image_paths)}\n")
        f.write(f"实验时间: {config_info['timestamp']}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("【实验结果】\n")
        f.write(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"正确预测: {int(accuracy * len(test_labels))}/{len(test_labels)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 统计每个类别的准确率
        f.write("【各类别准确率】\n")
        from collections import defaultdict
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, true in zip(predicted_categories, test_labels):
            category_stats[true]['total'] += 1
            if pred == true:
                category_stats[true]['correct'] += 1
        
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            cat_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"{category:25s}: {cat_acc:.4f} ({stats['correct']}/{stats['total']})\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"详细结果已保存到: {results_txt}")
    
    # 保存预测结果
    predictions_file = os.path.join(result_folder, 'predictions.npy')
    np.save(predictions_file, np.array(predicted_categories))
    print(f"预测结果已保存到: {predictions_file}")


def run_experiment(config, feature='tiny_image', classifier='nearest_neighbor', 
                   cnn_model='resnet18'):
    """
    运行单个实验
    
    参数:
        config: ConfigLoader对象
        feature: 特征提取方法
        classifier: 分类器
        cnn_model: CNN模型名称
    
    返回:
        accuracy: 准确率
        result_folder: 结果文件夹路径
    """
    
    print("\n" + "=" * 80)
    print(f"实验: {feature} + {classifier}" + (f" ({cnn_model})" if feature.lower() == 'cnn' else ""))
    print("=" * 80)
    
    # 获取配置
    dataset_name = config.get_dataset_name()
    data_path = config.get_data_path()
    categories = config.get_categories()
    abbr_categories = config.get_abbr_categories()
    num_train_per_cat = config.get_num_train_per_cat()
    
    # 创建结果文件夹（使用时间戳确保不覆盖）
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_acc = '0.0000'
    feature_name = get_feature_name(feature, cnn_model)
    classifier_name = get_classifier_name(classifier)
    
    result_folder_name = f"{dataset_name}_{len(categories)}类_{feature_name}_{classifier_name}_{temp_acc}_{current_time}"
    results_config = config.get_results_config()
    result_folder_path = os.path.join(results_config['save_path'], result_folder_name)
    
    os.makedirs(result_folder_path, exist_ok=True)
    print(f'\n实验结果将保存到: {result_folder_path}')
    
    # 加载数据
    print('\n获取所有训练和测试数据的路径和标签...')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)
    
    print(f'\n使用 {feature} 特征表示图片。')
    
    # 提取特征
    if feature.lower() == 'tiny_image':
        print('提取Tiny Image特征...')
        feature_config = config.get_feature_config('tiny_image')
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)
        print('Tiny Image特征提取完成。')
    
    elif feature.lower() == 'bag_of_words':
        vocab_path = os.path.join(data_path, 'vocab.npy')
        feature_config = config.get_feature_config('bag_of_words')
        vocab_size = feature_config['vocab_size']
        
        if not os.path.isfile(vocab_path):
            print(f'构建视觉词汇表（大小: {vocab_size}）...')
            vocab = build_vocabulary(train_image_paths, vocab_size)
            np.save(vocab_path, vocab)
            print(f'视觉词汇表已保存到 {vocab_path}')
        
        print('提取Bag of Words特征...')
        train_image_feats = get_bags_of_words(train_image_paths, vocab_path)
        test_image_feats = get_bags_of_words(test_image_paths, vocab_path)
        print('Bag of Words特征提取完成。')
    
    elif feature.lower() == 'cnn':
        feature_config = config.get_feature_config('cnn')
        batch_size = feature_config['batch_size']
        use_gpu = feature_config['use_gpu']
        
        print(f'使用CNN模型: {cnn_model}')
        print('为训练图片提取CNN特征...')
        train_image_feats = get_cnn_features(train_image_paths, 
                                            model_name=cnn_model, 
                                            batch_size=batch_size, 
                                            use_gpu=use_gpu)
        print('为测试图片提取CNN特征...')
        test_image_feats = get_cnn_features(test_image_paths, 
                                           model_name=cnn_model, 
                                           batch_size=batch_size, 
                                           use_gpu=use_gpu)
        print('CNN特征提取完成。')
    
    else:
        raise ValueError(f'未知的特征类型: {feature}')
    
    # 分类
    print(f'\n使用 {classifier} 分类器预测测试集类别...')
    
    if classifier.lower() == 'nearest_neighbor':
        classifier_config = config.get_classifier_config('nearest_neighbor')
        k = classifier_config.get('k', 1)
        predicted_categories = nearest_neighbor_classify(train_image_feats, 
                                                        train_labels, 
                                                        test_image_feats, k=k)
    
    elif classifier.lower() == 'support_vector_machine':
        if not HAS_SVM:
            raise ValueError('SVM分类器不可用')
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
    
    else:
        raise ValueError(f'未知的分类器类型: {classifier}')
    
    # 计算准确率
    accuracy = calculate_accuracy(predicted_categories, test_labels)
    print('\n' + '=' * 80)
    print(f'分类准确率: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print('=' * 80)
    
    # 保存结果
    save_results_to_folder(result_folder_path, train_image_paths, test_image_paths,
                          train_labels, test_labels, predicted_categories,
                          feature, classifier, cnn_model, accuracy, dataset_name)
    
    # 重命名文件夹（包含准确率）
    final_acc = f"{accuracy:.4f}"
    new_folder_name = f"{dataset_name}_{len(categories)}类_{feature_name}_{classifier_name}_{final_acc}_{current_time}"
    new_folder_path = os.path.join(results_config['save_path'], new_folder_name)
    
    os.rename(result_folder_path, new_folder_path)
    result_folder_path = new_folder_path
    print(f'\n结果文件夹已更新为: {result_folder_path}')
    
    # 生成结果网页
    if results_config.get('create_webpage', True):
        print('\n生成结果网页...')
        webpage_dir = 'results_webpage_temp'
        create_results_webpage(train_image_paths,
                             test_image_paths,
                             train_labels,
                             test_labels,
                             categories,
                             abbr_categories,
                             predicted_categories,
                             output_dir=webpage_dir)
        
        # 复制到结果文件夹
        if os.path.exists(os.path.join(webpage_dir, 'index.html')):
            shutil.copy(os.path.join(webpage_dir, 'index.html'), 
                       os.path.join(result_folder_path, 'index.html'))
            print(f'结果网页已复制到: {os.path.join(result_folder_path, "index.html")}')
        
        if os.path.exists(os.path.join(webpage_dir, 'confusion_matrix.png')):
            shutil.copy(os.path.join(webpage_dir, 'confusion_matrix.png'), 
                       os.path.join(result_folder_path, 'confusion_matrix.png'))
            print(f'混淆矩阵已复制到: {os.path.join(result_folder_path, "confusion_matrix.png")}')
        
        # 复制缩略图文件夹
        if os.path.exists(os.path.join(webpage_dir, 'thumbnails')):
            thumbnails_dest = os.path.join(result_folder_path, 'thumbnails')
            if os.path.exists(thumbnails_dest):
                shutil.rmtree(thumbnails_dest)
            shutil.copytree(os.path.join(webpage_dir, 'thumbnails'), thumbnails_dest)
    
    print('\n' + '=' * 80)
    print('实验完成！')
    print(f'所有结果已保存到: {result_folder_path}')
    print('=' * 80 + '\n')
    
    return accuracy, result_folder_path


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='场景识别实验')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str, default=None,
                       help='数据集名称 (15scene, 67indoor, custom)')
    parser.add_argument('--feature', type=str, default='tiny_image',
                       choices=['tiny_image', 'bag_of_words', 'cnn'],
                       help='特征提取方法')
    parser.add_argument('--classifier', type=str, default='nearest_neighbor',
                       choices=['nearest_neighbor', 'support_vector_machine'],
                       help='分类器')
    parser.add_argument('--cnn-model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vgg16'],
                       help='CNN模型（当feature=cnn时使用）')
    parser.add_argument('--run-all', action='store_true',
                       help='运行所有实验组合')
    parser.add_argument('--list-datasets', action='store_true',
                       help='列出所有可用的数据集')
    
    args = parser.parse_args()
    
    # 加载配置
    print("=" * 80)
    print("加载配置文件...")
    print("=" * 80)
    
    config = load_config(args.config)
    
    # 列出数据集
    if args.list_datasets:
        print("\n可用的数据集:")
        datasets = config.get_available_datasets()
        for i, ds in enumerate(datasets, 1):
            try:
                config.set_dataset_name(ds)
                path = config.get_data_path()
                exists = "✓" if os.path.exists(path) else "✗"
                num_cat = len(config.get_categories())
                print(f"  {i}. {ds:15s} {exists} ({num_cat:2d} 类) {path}")
            except Exception as e:
                print(f"  {i}. {ds:15s} ✗ (配置错误: {e})")
        return
    
    # 如果指定了数据集，切换到该数据集
    if args.dataset:
        config.set_dataset_name(args.dataset)
    
    config.print_config_summary()
    
    # 验证配置
    is_valid, message = config.validate_dataset()
    if not is_valid:
        print(f"\n错误: {message}")
        sys.exit(1)
    
    print(f"\n配置验证通过: {message}\n")
    
    # 运行实验
    if args.run_all:
        print("=" * 80)
        print("运行所有实验组合")
        print("=" * 80)
        
        experiments = [
            ('tiny_image', 'nearest_neighbor', None),
            ('bag_of_words', 'nearest_neighbor', None),
            ('cnn', 'nearest_neighbor', 'resnet18'),
        ]
        
        # 如果有SVM，添加SVM实验
        if HAS_SVM:
            experiments.extend([
                ('bag_of_words', 'support_vector_machine', None),
                ('cnn', 'support_vector_machine', 'resnet18'),
            ])
        
        results = []
        for feature, classifier, cnn_model in experiments:
            try:
                accuracy, folder = run_experiment(config, feature, classifier, 
                                                 cnn_model or 'resnet18')
                results.append((feature, classifier, cnn_model, accuracy, folder))
            except Exception as e:
                print(f"\n实验失败: {feature} + {classifier}")
                print(f"错误: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        
        # 打印总结
        print("\n" + "=" * 80)
        print("所有实验完成！")
        print("=" * 80)
        for feature, classifier, cnn_model, accuracy, folder in results:
            model_str = f" ({cnn_model})" if cnn_model else ""
            print(f"{feature}{model_str} + {classifier}: {accuracy:.4f}")
        print("=" * 80)
    
    else:
        # 运行单个实验
        run_experiment(config, args.feature, args.classifier, args.cnn_model)


if __name__ == '__main__':
    main()

'''
# 在15scene数据集上运行所有实验组合
python main_improved.py --dataset 15scene --run-all
# 在67indoor数据集上运行所有实验组合
python main_improved.py --dataset 67indoor --run-all

单个测试方法：
# === 组合3: Bag of Words + SVM ===
python main_improved.py --dataset 15scene --feature bag_of_words --classifier support_vector_machine
'''
