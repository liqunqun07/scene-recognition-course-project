"""
辅助函数
用于获取图像路径和标签
"""
import os
import glob
import warnings

warnings.filterwarnings('ignore')


def get_image_paths(data_path, categories, num_train_per_cat):
    """
    获取训练和测试图像的路径和标签
    
    参数:
        data_path: 数据集根路径
        categories: 类别列表
        num_train_per_cat: 每个类别的最大训练样本数
    
    返回:
        train_image_paths: 训练图像路径列表
        test_image_paths: 测试图像路径列表
        train_labels: 训练标签列表
        test_labels: 测试标签列表
    """
    
    num_categories = len(categories)
    
    # 初始化列表
    train_image_paths = []
    test_image_paths = []
    train_labels = []
    test_labels = []
    
    print("=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    # 统计信息
    total_train = 0
    total_test = 0
    missing_categories = []
    
    for i, cat in enumerate(categories):
        # 获取训练集图像
        train_cat_path = os.path.join(data_path, 'train', cat)
        
        if not os.path.exists(train_cat_path):
            print(f"警告: 训练类别路径不存在: {train_cat_path}")
            missing_categories.append(cat)
            continue
        
        # 支持多种图像格式
        train_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']:
            train_images.extend(glob.glob(os.path.join(train_cat_path, ext)))
        
        # 去重并排序
        train_images = sorted(list(set(train_images)))
        
        # 限制数量
        actual_train_num = min(num_train_per_cat, len(train_images))
        
        if actual_train_num == 0:
            print(f"警告: 类别 {cat} 没有训练图像")
            continue
        
        # 添加训练图像和标签
        for j in range(actual_train_num):
            train_image_paths.append(train_images[j])
            train_labels.append(cat)
        
        total_train += actual_train_num
        
        # 获取测试集图像
        test_cat_path = os.path.join(data_path, 'test', cat)
        
        if not os.path.exists(test_cat_path):
            print(f"警告: 测试类别路径不存在: {test_cat_path}")
            continue
        
        # 支持多种图像格式
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']:
            test_images.extend(glob.glob(os.path.join(test_cat_path, ext)))
        
        # 去重并排序
        test_images = sorted(list(set(test_images)))
        
        # 限制数量
        actual_test_num = min(num_train_per_cat, len(test_images))
        
        if actual_test_num == 0:
            print(f"警告: 类别 {cat} 没有测试图像")
            continue
        
        # 添加测试图像和标签
        for j in range(actual_test_num):
            test_image_paths.append(test_images[j])
            test_labels.append(cat)
        
        total_test += actual_test_num
        
        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"已加载 {i + 1}/{num_categories} 个类别")
    
    # 打印统计信息
    print("=" * 80)
    print(f"数据集加载完成")
    print(f"训练图像: {total_train} 张")
    print(f"测试图像: {total_test} 张")
    print(f"总类别数: {len(categories)}")
    
    if missing_categories:
        print(f"\n缺失的类别 ({len(missing_categories)}):")
        for cat in missing_categories:
            print(f"  - {cat}")
    
    print("=" * 80)
    
    # 验证数据
    if len(train_image_paths) == 0:
        raise ValueError("错误: 没有找到训练图像！请检查数据路径和类别名称。")
    
    if len(test_image_paths) == 0:
        raise ValueError("错误: 没有找到测试图像！请检查数据路径和类别名称。")
    
    return (train_image_paths, test_image_paths, train_labels, test_labels)


def validate_data_structure(data_path, categories):
    """
    验证数据集结构是否正确
    
    参数:
        data_path: 数据集根路径
        categories: 类别列表
    
    返回:
        valid: 是否有效
        issues: 问题列表
    """
    issues = []
    
    # 检查根路径
    if not os.path.exists(data_path):
        issues.append(f"数据根路径不存在: {data_path}")
        return False, issues
    
    # 检查train和test文件夹
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    if not os.path.exists(train_path):
        issues.append(f"训练数据路径不存在: {train_path}")
    
    if not os.path.exists(test_path):
        issues.append(f"测试数据路径不存在: {test_path}")
    
    # 检查每个类别
    for cat in categories:
        train_cat_path = os.path.join(train_path, cat)
        test_cat_path = os.path.join(test_path, cat)
        
        if not os.path.exists(train_cat_path):
            issues.append(f"训练类别路径不存在: {train_cat_path}")
        
        if not os.path.exists(test_cat_path):
            issues.append(f"测试类别路径不存在: {test_cat_path}")
    
    valid = len(issues) == 0
    return valid, issues


def print_data_structure(data_path, categories):
    """
    打印数据集结构信息
    
    参数:
        data_path: 数据集根路径
        categories: 类别列表
    """
    print("=" * 80)
    print("数据集结构信息")
    print("=" * 80)
    print(f"数据路径: {data_path}")
    print(f"类别数量: {len(categories)}")
    print("\n类别列表:")
    for i, cat in enumerate(categories):
        print(f"  {i+1:2d}. {cat}")
    print("=" * 80)

