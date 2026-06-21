"""
图像预处理工具
用于统一不同数据集的图像格式，处理各种图像问题
"""
import os
import glob
import numpy as np
from skimage import io
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
from skimage.transform import resize
from PIL import Image
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config):
        """
        初始化预处理器
        
        参数:
            config: 预处理配置字典
        """
        self.enabled = config.get('enabled', True)
        self.output_path = config.get('output_path', './preprocessed_data/')
        self.target_format = config.get('target_format', 'jpg')
        self.resize_shape = config.get('resize', None)
        self.convert_to_rgb = config.get('convert_to_rgb', True)
        self.keep_structure = config.get('keep_structure', True)
    
    def process_single_image(self, image_path, output_path=None):
        """
        处理单张图像
        
        参数:
            image_path: 输入图像路径
            output_path: 输出图像路径，如果为None则自动生成
        
        返回:
            output_path: 处理后的图像路径
            success: 是否成功
        """
        try:
            # 读取图像
            image = self._safe_read_image(image_path)
            
            if image is None:
                return None, False
            
            # 处理图像维度
            image = self._normalize_dimensions(image)
            
            # 转换为RGB
            if self.convert_to_rgb:
                image = self._convert_to_rgb(image)
            
            # 调整大小
            if self.resize_shape is not None:
                image = self._resize_image(image, self.resize_shape)
            
            # 生成输出路径
            if output_path is None:
                output_path = self._generate_output_path(image_path)
            
            # 保存图像
            self._save_image(image, output_path)
            
            return output_path, True
            
        except Exception as e:
            print(f"处理图像失败 {image_path}: {str(e)}")
            return None, False
    
    def process_dataset(self, data_path, categories):
        """
        处理整个数据集
        
        参数:
            data_path: 数据集根路径
            categories: 类别列表
        
        返回:
            stats: 处理统计信息
        """
        print("=" * 80)
        print("开始预处理数据集")
        print("=" * 80)
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        # 遍历train和test
        for split in ['train', 'test']:
            split_path = os.path.join(data_path, split)
            
            if not os.path.exists(split_path):
                print(f"警告: 路径不存在: {split_path}")
                continue
            
            print(f"\n处理 {split} 数据...")
            
            # 遍历类别
            for category in tqdm(categories, desc=f"处理{split}集"):
                cat_path = os.path.join(split_path, category)
                
                if not os.path.exists(cat_path):
                    print(f"警告: 类别路径不存在: {cat_path}")
                    continue
                
                # 获取所有图像
                image_files = glob.glob(os.path.join(cat_path, '*.*'))
                image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                
                # 处理每张图像
                for img_path in image_files:
                    stats['total'] += 1
                    _, success = self.process_single_image(img_path)
                    
                    if success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                        stats['failed_files'].append(img_path)
        
        # 打印统计信息
        self._print_stats(stats)
        
        return stats
    
    def _safe_read_image(self, image_path):
        """
        安全地读取图像，处理各种格式问题
        
        参数:
            image_path: 图像路径
        
        返回:
            image: 图像数组，如果失败返回None
        """
        try:
            # 尝试用skimage读取
            image = io.imread(image_path)
            return image
        except Exception as e1:
            try:
                # 尝试用PIL读取
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                return image
            except Exception as e2:
                print(f"无法读取图像 {image_path}: skimage错误={e1}, PIL错误={e2}")
                return None
    
    def _normalize_dimensions(self, image):
        """
        标准化图像维度
        
        参数:
            image: 输入图像
        
        返回:
            normalized_image: 标准化后的图像
        """
        # 处理4D图像（批次维度）
        if len(image.shape) == 4:
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = image[0]
        
        # 处理单通道彩色图像（3D但第三维为1）
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        
        return image
    
    def _convert_to_rgb(self, image):
        """
        转换图像为RGB
        
        参数:
            image: 输入图像
        
        返回:
            rgb_image: RGB图像
        """
        # 已经是RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image
        
        # RGBA转RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            try:
                return rgba2rgb(image)
            except:
                # 简单方法：取前3个通道
                return image[:, :, :3]
        
        # 灰度图转RGB
        if len(image.shape) == 2:
            try:
                return gray2rgb(image)
            except:
                # 简单方法：复制3次
                return np.stack([image] * 3, axis=-1)
        
        # 其他情况：尝试取前3个通道或平均
        if len(image.shape) == 3:
            if image.shape[2] > 3:
                return image[:, :, :3]
            else:
                # 尝试转为灰度再转RGB
                gray = np.mean(image, axis=2)
                return np.stack([gray] * 3, axis=-1)
        
        return image
    
    def _resize_image(self, image, target_shape):
        """
        调整图像大小
        
        参数:
            image: 输入图像
            target_shape: 目标形状 [height, width]
        
        返回:
            resized_image: 调整后的图像
        """
        try:
            if len(image.shape) == 2:
                # 灰度图
                return resize(image, target_shape, anti_aliasing=True, preserve_range=True)
            else:
                # 彩色图
                return resize(image, target_shape, anti_aliasing=True, preserve_range=True)
        except Exception as e:
            print(f"调整图像大小失败: {e}")
            return image
    
    def _generate_output_path(self, input_path):
        """
        生成输出路径
        
        参数:
            input_path: 输入路径
        
        返回:
            output_path: 输出路径
        """
        if self.keep_structure:
            # 保持目录结构
            rel_path = os.path.relpath(input_path)
            output_path = os.path.join(self.output_path, rel_path)
        else:
            # 扁平化结构
            filename = os.path.basename(input_path)
            output_path = os.path.join(self.output_path, filename)
        
        # 更改扩展名
        output_path = os.path.splitext(output_path)[0] + f'.{self.target_format}'
        
        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        return output_path
    
    def _save_image(self, image, output_path):
        """
        保存图像
        
        参数:
            image: 图像数组
            output_path: 输出路径
        """
        # 确保图像是uint8类型
        if image.dtype != np.uint8:
            # 标准化到0-255
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 保存
        io.imsave(output_path, image, check_contrast=False)
    
    def _print_stats(self, stats):
        """打印统计信息"""
        print("\n" + "=" * 80)
        print("预处理完成")
        print("=" * 80)
        print(f"总图像数: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        
        if stats['failed'] > 0:
            print(f"\n失败率: {stats['failed']/stats['total']*100:.2f}%")
            print("\n失败的文件:")
            for f in stats['failed_files'][:10]:  # 只显示前10个
                print(f"  - {f}")
            if len(stats['failed_files']) > 10:
                print(f"  ... 还有 {len(stats['failed_files'])-10} 个")
        
        print("=" * 80)


def preprocess_images(config_loader):
    """
    预处理图像的便捷函数
    
    参数:
        config_loader: ConfigLoader对象
    
    返回:
        stats: 处理统计信息
    """
    # 获取配置
    preprocess_config = config_loader.get_preprocessing_config()
    
    if not preprocess_config['enabled']:
        print("预处理已禁用")
        return None
    
    # 创建预处理器
    preprocessor = ImagePreprocessor(preprocess_config)
    
    # 获取数据集信息
    data_path = config_loader.get_data_path()
    categories = config_loader.get_categories()
    
    # 处理数据集
    stats = preprocessor.process_dataset(data_path, categories)
    
    return stats



