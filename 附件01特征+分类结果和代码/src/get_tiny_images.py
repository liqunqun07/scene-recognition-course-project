import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
import warnings

warnings.filterwarnings('ignore')


def get_tiny_images(image_paths):
    '''
    This function returns a set of feature vectors for a set of images.
    '''

    # 设置tiny image的尺寸
    tiny_size = 16

    # 初始化特征矩阵
    num_images = len(image_paths)
    image_feats = np.zeros((num_images, tiny_size * tiny_size))

    for i, path in enumerate(image_paths):
        try:
            # 读取图像
            image = imread(path)

            # 调试信息（每100张图片显示一次）
            if i % 100 == 0:
                print(f"Processing image {i + 1}/{num_images}: {path}")
                print(f"  Shape: {image.shape}, dtype: {image.dtype}")

            # 处理4D图像（包含批次维度）
            if len(image.shape) == 4:
                # 形状为 (batch, height, width, channels)
                if image.shape[0] == 1:
                    image = image[0]  # 去掉批次维度
                else:
                    # 取第一张
                    image = image[0]

            # 处理不同通道数的图像
            if len(image.shape) == 3:
                # 3D图像 - 检查通道数
                if image.shape[2] == 4:
                    # RGBA图像 - 转换为RGB，然后转为灰度
                    try:
                        image_rgb = rgba2rgb(image)
                        image_gray = rgb2gray(image_rgb)
                        image = image_gray
                    except:
                        # 如果rgba2rgb失败，使用简单方法：取前3个通道
                        image = np.mean(image[:, :, :3], axis=2)
                elif image.shape[2] == 3:
                    # RGB图像 - 转换为灰度
                    try:
                        image = rgb2gray(image)
                    except:
                        # 如果rgb2gray失败，使用简单平均
                        image = np.mean(image, axis=2)
                else:
                    # 其他通道数 - 取第一个通道或平均
                    image = np.mean(image, axis=2)
            elif len(image.shape) == 2:
                # 已经是灰度图像
                pass
            else:
                print(f"Warning: Unexpected image shape {image.shape} for {path}")
                # 使用随机特征
                image_feats[i, :] = np.random.randn(tiny_size * tiny_size) * 0.1
                continue

            # 确保是2D图像
            if len(image.shape) != 2:
                print(f"Warning: Image still not 2D after processing: {image.shape}")
                # 强制转换为2D
                if len(image.shape) > 2:
                    image = np.mean(image, axis=2)
                else:
                    # 使用随机特征
                    image_feats[i, :] = np.random.randn(tiny_size * tiny_size) * 0.1
                    continue

            # 调整图像尺寸为tiny_size x tiny_size
            image_resized = resize(image, (tiny_size, tiny_size),
                                   anti_aliasing=True,
                                   preserve_range=True)

            # 标准化（零均值和单位方差）
            mean_val = np.mean(image_resized)
            std_val = np.std(image_resized)

            if std_val > 0.001:  # 避免除以非常小的数
                image_resized = (image_resized - mean_val) / std_val
            else:
                # 如果标准差太小，只进行零均值化
                image_resized = image_resized - mean_val

            # 展平为一维向量
            image_feats[i, :] = image_resized.flatten()

        except Exception as e:
            print(f"Error processing image {path}: {e}")
            print(f"  Shape: {image.shape if 'image' in locals() else 'N/A'}")
            # 使用随机值作为占位符（比零向量更好）
            image_feats[i, :] = np.random.randn(tiny_size * tiny_size) * 0.1

    return image_feats