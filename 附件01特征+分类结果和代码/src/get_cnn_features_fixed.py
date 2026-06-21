import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

'''
使用预训练的CNN模型提取图像特征
模型来自 PyTorch的torchvision库
模型是在ImageNet数据集上预训练好的
'''

class ImageDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # 读取图像
            image = imread(self.image_paths[idx])
            
            # 处理图像维度
            image = self._process_image(image)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"\n读取图像失败 {self.image_paths[idx]}: {e}")
            # 返回一个黑色图像作为占位符
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image
    
    def _process_image(self, image):
        """
        处理图像，确保返回RGB格式
        
        参数:
            image: 输入图像
        
        返回:
            rgb_image: RGB图像 (H, W, 3)
        """
        # 处理4D图像（batch维度）
        if len(image.shape) == 4:
            if image.shape[0] == 1:
                image = image[0]
            else:
                image = image[0]
        
        # 转换为RGB
        if len(image.shape) == 2:
            # 灰度图 -> RGB
            try:
                image = gray2rgb(image)
            except:
                image = np.stack([image] * 3, axis=-1)
        
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA -> RGB
                try:
                    image = rgba2rgb(image)
                except:
                    image = image[:, :, :3]
            elif image.shape[2] == 1:
                # 单通道 -> RGB
                image = np.stack([image[:, :, 0]] * 3, axis=-1)
            elif image.shape[2] == 3:
                # 已经是RGB
                pass
            else:
                # 其他情况：取前3个通道或复制第一个通道
                if image.shape[2] > 3:
                    image = image[:, :, :3]
                else:
                    image = np.stack([image[:, :, 0]] * 3, axis=-1)
        
        # 确保是uint8类型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 最终检查
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"警告: 图像处理后形状异常 {image.shape}，创建默认RGB图像")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        return image


def get_cnn_features(image_paths, model_name='resnet18', batch_size=32, use_gpu=True):
    '''
    使用预训练的CNN模型提取图像特征
    
    参数:
        image_paths: 图像路径列表
        model_name: 使用的CNN模型，可选 'resnet18', 'resnet50', 'vgg16'
        batch_size: 批处理大小
        use_gpu: 是否使用GPU加速
    
    返回:
        all_features: 特征矩阵 (n_images x feature_dim)
        - ResNet18: 512维
        - ResNet50: 2048维
        - VGG16: 4096维
    '''
    
    # 检查是否可以使用GPU
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'使用GPU: {torch.cuda.get_device_name(0)}')
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('使用Apple Silicon GPU (MPS)')
    else:
        device = torch.device('cpu')
        print('使用CPU')
    
    # 图像预处理变换（与ImageNet预训练模型一致）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0, 
                          pin_memory=True if device.type == 'cuda' else False)
    
    # 加载预训练模型
    print(f'加载预训练 {model_name} 模型...')
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # 移除最后的全连接层
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 512
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 2048
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # VGG使用分类器的倒数第二层
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        feature_dim = 4096
        
    else:
        raise ValueError(f'未知的模型: {model_name}. 可选: resnet18, resnet50, vgg16')
    
    # 将模型设置为评估模式并移动到指定设备
    model = model.to(device)
    model.eval()
    
    # 提取特征
    features_list = []
    print(f'从 {len(image_paths)} 张图像提取特征...')
    
    with torch.no_grad():  # 不需要计算梯度
        for i, batch in enumerate(tqdm(dataloader, desc=f"提取{model_name}特征")):
            try:
                batch = batch.to(device)
                features = model(batch)
                
                # 展平特征（如果是ResNet，需要squeeze掉空间维度）
                features = features.view(features.size(0), -1)
                
                # 移回CPU并转为numpy
                features = features.cpu().numpy()
                features_list.append(features)
                
            except Exception as e:
                print(f"\n处理批次 {i} 时出错: {e}")
                # 使用零向量作为占位符
                dummy_features = np.zeros((batch.size(0), feature_dim))
                features_list.append(dummy_features)
                continue
    
    # 合并所有特征
    all_features = np.vstack(features_list)
    
    # L2归一化（通常能提升性能）
    print("对特征进行L2归一化...")
    norms = np.linalg.norm(all_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # 避免除以0
    all_features = all_features / norms
    
    print(f'特征提取完成。形状: {all_features.shape}')
    return all_features


