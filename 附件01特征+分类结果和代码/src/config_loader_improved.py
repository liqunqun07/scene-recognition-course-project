"""
配置加载器 - 改进版
用于加载和管理项目配置
支持根据数据集名称自动选择数据路径
"""
import yaml
import os
from typing import Dict, List, Tuple, Any


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化配置加载器
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_dataset_name(self) -> str:
        """获取数据集名称"""
        return self.config['dataset']['name']
    
    def set_dataset_name(self, dataset_name: str):
        """
        设置数据集名称（运行时修改）
        
        参数:
            dataset_name: 数据集名称
        """
        if dataset_name not in self.config.get('dataset_paths', {}):
            available = list(self.config.get('dataset_paths', {}).keys())
            raise ValueError(f"未知的数据集: {dataset_name}. 可用的数据集: {available}")
        
        self.config['dataset']['name'] = dataset_name
        print(f"数据集已切换为: {dataset_name}")
        print(f"数据路径: {self.get_data_path()}")
    
    def get_data_path(self) -> str:
        """
        获取数据集路径（根据当前数据集名称自动选择）
        
        返回:
            数据集路径
        """
        dataset_name = self.get_dataset_name()
        
        # 从dataset_paths映射中获取路径
        data_path = None
        if 'dataset_paths' in self.config:
            if dataset_name in self.config['dataset_paths']:
                data_path = self.config['dataset_paths'][dataset_name]
        
        # 兼容旧配置格式（如果没有dataset_paths）
        if data_path is None and 'data_path' in self.config['dataset']:
            data_path = self.config['dataset']['data_path']

        if data_path is not None:
            data_path = os.path.expanduser(os.path.expandvars(str(data_path)))
            if not os.path.isabs(data_path):
                config_dir = os.path.dirname(os.path.abspath(self.config_path))
                data_path = os.path.abspath(os.path.join(config_dir, data_path))
            return data_path
        
        raise ValueError(f"未找到数据集 '{dataset_name}' 的路径配置")
    
    def get_num_train_per_cat(self) -> int:
        """获取每个类别的训练样本数"""
        return self.config['dataset']['num_train_per_cat']
    
    def get_categories(self, dataset_name: str = None) -> List[str]:
        """
        获取类别列表
        
        参数:
            dataset_name: 数据集名称，如果为None则使用配置中的默认数据集
        
        返回:
            类别列表
        """
        if dataset_name is None:
            dataset_name = self.get_dataset_name()
        
        if dataset_name not in self.config:
            raise ValueError(f"未知的数据集: {dataset_name}")
        
        return self.config[dataset_name]['categories']
    
    def get_abbr_categories(self, dataset_name: str = None) -> List[str]:
        """
        获取缩写类别列表
        
        参数:
            dataset_name: 数据集名称，如果为None则使用配置中的默认数据集
        
        返回:
            缩写类别列表
        """
        if dataset_name is None:
            dataset_name = self.get_dataset_name()
        
        if dataset_name not in self.config:
            raise ValueError(f"未知的数据集: {dataset_name}")
        
        return self.config[dataset_name]['abbr_categories']
    
    def get_preprocessing_config(self) -> Dict:
        """获取预处理配置"""
        return self.config['dataset']['preprocessing']
    
    def get_feature_config(self, feature_type: str) -> Dict:
        """
        获取特征提取配置
        
        参数:
            feature_type: 特征类型 (tiny_image, bag_of_words, cnn)
        
        返回:
            特征配置字典
        """
        if feature_type not in self.config['features']:
            raise ValueError(f"未知的特征类型: {feature_type}")
        
        return self.config['features'][feature_type]
    
    def get_classifier_config(self, classifier_type: str) -> Dict:
        """
        获取分类器配置
        
        参数:
            classifier_type: 分类器类型 (nearest_neighbor, svm)
        
        返回:
            分类器配置字典
        """
        if classifier_type not in self.config['classifiers']:
            raise ValueError(f"未知的分类器类型: {classifier_type}")
        
        return self.config['classifiers'][classifier_type]
    
    def get_results_config(self) -> Dict:
        """获取结果保存配置"""
        return self.config['results']
    
    def get_available_datasets(self) -> List[str]:
        """
        获取所有可用的数据集列表
        
        返回:
            数据集名称列表
        """
        if 'dataset_paths' in self.config:
            return list(self.config['dataset_paths'].keys())
        else:
            # 兼容旧版本：返回配置文件中定义的数据集
            datasets = []
            for key in self.config.keys():
                if key not in ['dataset', 'features', 'classifiers', 'results', 'dataset_paths']:
                    if 'categories' in self.config[key]:
                        datasets.append(key)
            return datasets
    
    def validate_dataset(self) -> Tuple[bool, str]:
        """
        验证数据集配置是否有效
        
        返回:
            (是否有效, 错误信息)
        """
        dataset_name = self.get_dataset_name()
        
        # 检查数据集是否存在
        if dataset_name not in self.config:
            return False, f"数据集 '{dataset_name}' 未在配置文件中定义"
        
        # 检查类别和缩写是否长度一致
        categories = self.get_categories()
        abbr_categories = self.get_abbr_categories()
        
        if len(categories) != len(abbr_categories):
            return False, f"类别数量({len(categories)})与缩写数量({len(abbr_categories)})不匹配"
        
        # 检查数据路径是否存在
        try:
            data_path = self.get_data_path()
            if not os.path.exists(data_path):
                return False, f"数据路径不存在: {data_path}"
        except ValueError as e:
            return False, str(e)
        
        return True, "配置验证通过"
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("=" * 80)
        print("配置摘要")
        print("=" * 80)
        print(f"数据集名称: {self.get_dataset_name()}")
        print(f"数据路径: {self.get_data_path()}")
        print(f"类别数量: {len(self.get_categories())}")
        print(f"每类训练样本数: {self.get_num_train_per_cat()}")
        print(f"预处理: {'启用' if self.get_preprocessing_config()['enabled'] else '禁用'}")
        
        # 显示所有可用数据集
        available_datasets = self.get_available_datasets()
        print(f"\n可用数据集: {', '.join(available_datasets)}")
        print("=" * 80)


def load_config(config_path='config.yaml') -> ConfigLoader:
    """
    便捷函数：加载配置
    
    参数:
        config_path: 配置文件路径
    
    返回:
        ConfigLoader对象
    """
    return ConfigLoader(config_path)
