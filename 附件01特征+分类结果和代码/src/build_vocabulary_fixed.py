import numpy as np
from skimage.io import imread
from skimage.feature import hog
from skimage.color import rgb2gray, rgba2rgb
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

'''
构建视觉词汇表
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


def build_vocabulary(image_paths, vocab_size):
    '''
    构建视觉词汇表
    
    参数:
        image_paths: 图像路径列表
        vocab_size: 词汇表大小
    
    返回:
        vocabulary: 视觉词汇表 (vocab_size x feature_dim)
    '''
    
    print(f"开始构建词汇表，处理 {len(image_paths)} 张图像...")
    
    # HOG参数
    cells_per_block = (2, 2)
    pixels_per_cell = (4, 4)
    t = cells_per_block[0]
    
    images_feature_vectors = []
    failed_count = 0
    
    # 使用进度条
    for i, image_path in enumerate(tqdm(image_paths, desc="提取HOG特征")):
        # 安全读取和处理图像
        image = safe_read_and_process_image(image_path)
        
        if image is None:
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
            
            images_feature_vectors.append(feature_vector)
            
        except Exception as e:
            print(f"\n提取HOG特征失败 {image_path}: {e}")
            failed_count += 1
            continue
    
    if len(images_feature_vectors) == 0:
        raise ValueError("错误: 没有成功提取任何特征！请检查图像路径和格式。")
    
    print(f"\n成功提取 {len(images_feature_vectors)} 张图像的特征")
    if failed_count > 0:
        print(f"失败: {failed_count} 张")
    
    # 合并所有特征
    images_feature_vectors = np.vstack(images_feature_vectors)
    print(f"特征矩阵形状: {images_feature_vectors.shape}")
    
    # 使用K-Means聚类
    print(f"\n使用MiniBatchKMeans聚类成 {vocab_size} 个视觉词...")
    t0 = time.time()
    
    k_means = MiniBatchKMeans(
        n_clusters=vocab_size, 
        max_iter=500,
        random_state=42,
        verbose=0
    ).fit(images_feature_vectors)
    
    print(f'聚类完成，耗时: {time.time() - t0:.2f}秒')
    
    vocabulary = np.vstack(k_means.cluster_centers_)
    print(f"词汇表形状: {vocabulary.shape}")
    
    return vocabulary


