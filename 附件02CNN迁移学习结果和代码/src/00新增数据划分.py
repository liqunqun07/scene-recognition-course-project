import cv2 as cv
import matplotlib
import os
import warnings
warnings.filterwarnings("ignore")
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'MacOSX'
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Songti SC', 'STFangsong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
project_root = os.getcwd()  # 获取当前工作目录
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import shutil
print(f"当前工作目录: {project_root}")
target_dir = os.environ.get("CNN_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.chdir(target_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集文件夹路径
dataset_dir = os.environ.get("SCENE_DATA_PATH", os.path.join(target_dir, "data", "15scene"))
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'test')
# 创建验证集文件夹
val_path = os.path.join(dataset_dir, 'val')
os.makedirs(val_path, exist_ok=True)


# 从训练集的每个类别中抽取20%作为验证集
print("正在划分验证集...")
for class_name in os.listdir(test_path):
    class_test_path = os.path.join(test_path, class_name)
    class_val_path = os.path.join(val_path, class_name)
    if os.path.isdir(class_test_path):
        os.makedirs(class_val_path, exist_ok=True)
        # 获取该类别所有图像
        images = [f for f in os.listdir(class_test_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(images) > 0:
            # 划分训练集和验证集
            test_imgs, val_imgs = train_test_split(
                images,
                test_size=0.2,  # 20%作为验证集
                random_state=42,  # 固定随机种子确保可重复
                shuffle=True
            )

            # 将验证集图像移动到验证集文件夹
            for img in val_imgs:
                src = os.path.join(class_test_path, img)
                dst = os.path.join(class_val_path, img)
                shutil.move(src, dst)
            print(f"类别 {class_name}: 测试集 {len(test_imgs)} 张, 验证集 {len(val_imgs)} 张")

print("验证集划分完成!")
