import numpy as np
from skimage.io import imread
from skimage.feature import hog
from skimage.color import rgb2gray, rgba2rgb
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

'''
词袋模型特征提取
'''

def safe_read_and_process_image(image_path):
    """
    安全地读取和处理图像，确保返回灰度图
    
    参数:
        image_path: 图像路径
    
    返回:
        image: 灰度图像，如果失败返回None
    """
    try:
        # 读取图像
        image = imread(image_path)
        
        # 处理4D图像（batch维度）
        if len(image.shape) == 4:
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = image[0]
        
        # 处理不同通道数的图像
        if len(image.shape) == 3:
            # 彩色图像
            if image.shape[2] == 4:
                # RGBA -> RGB -> 灰度
                try:
                    image = rgba2rgb(image)
                    image = rgb2gray(image)
                except:
                    # 简单方法：取前3个通道再转灰度
                    image = np.mean(image[:, :, :3], axis=2)
            elif image.shape[2] == 3:
                # RGB -> 灰度
                try:
                    image = rgb2gray(image)
                except:
                    # 简单平均
                    image = np.mean(image, axis=2)
            elif image.shape[2] == 1:
                # 单通道
                image = image[:, :, 0]
            else:
                # 其他情况：平均所有通道
                image = np.mean(image, axis=2)
        
        elif len(image.shape) == 2:
            # 已经是灰度图
            pass
        else:
            print(f"警告: 图像维度异常 {image.shape} for {image_path}")
            return None
        
        # 确保是2D灰度图
        if len(image.shape) != 2:
            print(f"警告: 处理后图像仍不是2D {image.shape} for {image_path}")
            return None
        
        # 标准化到0-1
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
        
    except Exception as e:
        print(f"读取图像失败 {image_path}: {e}")
        return None


def get_bags_of_words(image_paths, vocab_path='vocab.npy'):
    '''
    使用词袋模型提取特征
    
    参数:
        image_paths: 图像路径列表
        vocab_path: 词汇表文件路径
    
    返回:
        images_histograms: Bag of Words特征 (n_images x vocab_size)
    
    步骤:
    1. 从图片提取很多局部特征（HOG）
    2. 每个局部特征在词汇表里找最相似的"视觉单词"
    3. 统计每个"视觉单词"出现的次数，做成直方图
    '''
    
    # 加载词汇表
    try:
        vocab = np.load(vocab_path)
        print(f'已加载词汇表: {vocab.shape}')
    except Exception as e:
        print(f"错误: 无法加载词汇表 {vocab_path}")
        print(f"请先运行 build_vocabulary 构建词汇表")
        raise e
    
    # HOG参数（必须与构建词汇表时相同）
    cells_per_block = (2, 2)
    pixels_per_cell = (4, 4)
    t = cells_per_block[0]
    
    images_histograms = []
    failed_count = 0
    
    print(f"提取 {len(image_paths)} 张图像的Bag of Words特征...")
    
    # 使用进度条
    for i, image_path in enumerate(tqdm(image_paths, desc="提取BoW特征")):
        # 安全读取和处理图像
        image = safe_read_and_process_image(image_path)
        
        if image is None:
            # 使用零向量作为占位符
            histogram = np.zeros(len(vocab))
            images_histograms.append(histogram)
            failed_count += 1
            continue
        
        try:
            # 提取HOG特征
            # channel_axis=None 明确指定这是灰度图
            feature_vector = hog(
                image, 
                feature_vector=True, 
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                channel_axis=None,  # 重要：指定为灰度图
                visualize=False
            ).reshape(-1, t*t*9)
            
            # 计算当前图片的feature与词袋的距离
            dist = cdist(vocab, feature_vector, metric='euclidean')
            
            # 选择最短距离，计算直方图
            min_dis_index = np.argmin(dist, axis=0)
            histogram, bin_edges = np.histogram(min_dis_index, bins=len(vocab))
            
            # 归一化
            if norm(histogram) > 0:
                histogram = histogram / norm(histogram)
            
            images_histograms.append(histogram)
            
        except Exception as e:
            print(f"\n提取特征失败 {image_path}: {e}")
            # 使用零向量作为占位符
            histogram = np.zeros(len(vocab))
            images_histograms.append(histogram)
            failed_count += 1
            continue
    
    print(f"\n成功提取 {len(image_paths) - failed_count} 张图像的特征")
    if failed_count > 0:
        print(f"失败/跳过: {failed_count} 张")
    
    return np.array(images_histograms)

