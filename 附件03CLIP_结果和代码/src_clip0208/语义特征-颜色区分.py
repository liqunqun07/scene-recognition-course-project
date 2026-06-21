import cv2 as cv
import matplotlib
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Songti SC', 'STFangsong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
project_root = os.getcwd()
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
target_dir = os.environ.get("CLIP_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
from transformers import CLIPModel, CLIPProcessor
os.chdir(target_dir)
from create_results_webpage import create_results_webpage_simple
import config_clip as config  # 使用CLIP专用配置
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
#加载模型
model_name = config.CLIP_MODEL_NAME
local_model_path = config.LOCAL_MODEL_PATH

if os.path.exists(local_model_path):
    print(f"从本地加载CLIP模型: {local_model_path}")
    model = CLIPModel.from_pretrained(local_model_path)
    processor = CLIPProcessor.from_pretrained(local_model_path)
else:
    print(f"下载CLIP模型: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    os.makedirs(local_model_path, exist_ok=True)
    model.save_pretrained(local_model_path)
    processor.save_pretrained(local_model_path)
    print(f"模型已保存到: {local_model_path}")
best_model_path = os.environ.get(
    "CLIP_BEST_MODEL_PATH",
    os.path.join(target_dir, "57_CLIP_clip_vit_base_patch16", "best_model.pth")
)
model.load_state_dict(torch.load(best_model_path))
model.eval().to(device)

#载入测试集图像
#step1 准备数据--------------------------------------------
result_path = os.environ.get("CLIP_RESULT_PATH", str(Path(best_model_path).parent))
df = pd.read_csv(os.path.join(result_path,'测试集预测结果.csv'))
df.head()
classes = df['标注类别名称'].unique()
print(classes)
# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

#step2 定义抽取方式-------------------------------------------------

encoding_array = []
img_path_list = []

for img_path in tqdm(df['图像路径']):
    img_path_list.append(img_path)
    img_pil = Image.open(img_path).convert('RGB')

    # 使用 CLIP processor 处理图像（替代 test_transform）
    inputs = processor(images=img_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        # 方法1: 提取最后一层的 CLS token（pooler_output）
         vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
         feature = vision_outputs.pooler_output.squeeze().cpu().numpy()  # [768] 对于 ViT-B/32
        # 方法2: 如果想要经过投影层的特征（维度更低，512）
        #feature = model.get_image_features(pixel_values=inputs.pixel_values).squeeze().cpu().numpy()

    encoding_array.append(feature)

encoding_array = np.array(encoding_array)
print(f"✅ 特征提取完成！特征形状: {encoding_array.shape}")

# 保存为本地的 npy 文件
np.save('{}_CLIP测试集语义特征.npy'.format(len(classes)), encoding_array)
print(f"✅ 特征已保存: {len(classes)}_project_CLIP测试集语义特征.npy")

# ----------设置可视化配色---------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import umap
import umap.plot
import plotly.express as px

# ======  新增场景分组定义 ======
# 将57个类别按属性分为7个自然语义组
scene_categories = {
    '自然景观': ['beach', 'desert', 'forest', 'mountain', 'ocean', 'valley', 'volcano',
                 'waterfall', 'canyon', 'cliff', 'glacier', 'reef', 'river', 'lake',
                 'island', 'cave','harbor','fountain','zoo', 'botanical garden'],

    '人文建筑': ['castle', 'temple', 'ancient temple', 'church', 'monument',
                 'bridge', 'lighthouse', 'windmill'],

    '文化场馆': ['museum', 'art gallery', 'exhibition hall', 'library', 'theater', 'concert hall','opera house'],

    '城市生活': ['city walk', 'entertainment district', 'nightclub', 'shopping mall',
                 'souvenir shop', 'dining', 'cafe', 'restaurant', 'market', 'park', 'Citizen Park'],

    '旅游活动': ['hiking', 'skiing', 'cable car',
                'camping','sunbathing', 'picnic'],

    '人群和拍照': ['selfie','solo traveler', 'backpacker', 'senior traveler', 'family with kids', 'couple','Group photo','tour group'],

    '交通方式':['railway', 'train','airport']

}

# ====== 为每个分组定义主色调 ======
# 每个场景组一个基础色，便于视觉上识别聚类
group_colors = {
    '自然景观': '#2E8B57',  # 海洋绿
    '人文建筑': '#8B4513',  # 棕色
    '文化场馆': '#4169E1',  # 皇家蓝
    '城市生活': '#FF6347',  # 番茄红
    '旅游活动': '#FFA500',  # 橙色
    '人群和拍照': '#9370DB',  # 中紫色
    '交通方式': '#708090'       # 石板灰
}


# ====== 新增颜色生成函数 ======
# 为每个类别分配颜色（在同组内使用相近色，但略有区分）
def get_scene_color(category):
    # 查找类别所属的分组
    for group, items in scene_categories.items():
        if category in items:
            base_color = group_colors[group]
            # 在组内生成不同的色调
            group_index = items.index(category)
            # 轻微调整色调，使同组内可区分
            if group == '自然景观':
                colors = sns.light_palette(base_color, n_colors=len(items) + 2, reverse=True)[2:]
            elif group == '人文建筑':
                colors = sns.dark_palette(base_color, n_colors=len(items) + 2)[2:]
            else:
                colors = sns.color_palette(f"light:{base_color}", n_colors=len(items))
            return colors[group_index]
    return '#808080'  # 默认灰色


# ====== 生成颜色映射 ======
# 基于语义分组分配颜色
class_list = np.unique(df['标注类别名称'])
color_dict = {category: get_scene_color(category) for category in class_list}

# 设置点型列表
marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8',
               's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

# 随机打乱点型（保持可重复性）
random.seed(1234)
random.shuffle(marker_list)

# ----------STEP4 UMAP降维---------------------------
mapper = umap.UMAP(n_neighbors=10, n_components=2, random_state=12).fit(encoding_array)
X_umap_2d = mapper.embedding_

# ======  增强静态可视化 ======
plt.figure(figsize=(30, 23))
for idx, category in enumerate(class_list):
    color = color_dict[category]
    marker = marker_list[idx % len(marker_list)]
    indices = np.where(df['标注类别名称'] == category)
    plt.scatter(X_umap_2d[indices, 0], X_umap_2d[indices, 1],
                color=color, marker=marker, label=category, s=400, alpha=0.7)

# ====== 增强图例和标题 ======

plt.legend(fontsize=12, markerscale=1, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks([])
plt.yticks([])
plt.title('57类语义特征UMAP二维降维可视化\n(颜色分组: 自然景观/人文建筑/文化场馆/城市生活/旅游活动/人群类型/特殊地点)',
          fontsize=20, pad=20)
plt.tight_layout()
plt.savefig('57类语义特征UMAP二维降维可视化_分组配色.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ----------STEP5 交互式可视化（增强版）---------------------------
df_2d = pd.DataFrame()
df_2d['X'] = X_umap_2d[:, 0]
df_2d['Y'] = X_umap_2d[:, 1]
df_2d['标注类别名称'] = df['标注类别名称']
df_2d['预测类别'] = df['top-1-预测名称']
df_2d['图像路径'] = df['图像路径']


# ======  新增分组信息列 ======
# 为每个样本添加场景分组，便于交互式可视化分组着色
def get_category_group(category):
    for group, items in scene_categories.items():
        if category in items:
            return group
    return '其他'


df_2d['场景分组'] = df_2d['标注类别名称'].apply(get_category_group)

# 保存数据
df_2d.to_csv('57类_UMAP-2D_分组信息.csv', index=False)

# ======增强交互式可视化 ======

# color='场景分组'（按分组着色），便于观察聚类效果
fig = px.scatter(df_2d,
                 x='X',
                 y='Y',
                 color='场景分组',  # 改为按分组着色，更清晰显示聚类
                 symbol='标注类别名称',  # 保留具体类别符号
                 hover_data=['标注类别名称', '预测类别'],
                 hover_name='图像路径',
                 opacity=0.8,
                 width=1200,
                 height=800,
                 title='57类语义特征UMAP降维可视化（按场景分组）',
                 color_discrete_map=group_colors  # 使用定义的分组颜色
                 )

# 优化布局
fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05,
        font=dict(size=15)
    ),
    margin=dict(l=0, r=200, b=0, t=50)
)

# 显示和保存
fig.show()
fig.write_html('57类_语义特征UMAP二维降维_交互式分组.html')

# ====== 统计输出 ======
# 打印分组统计信息，便于验证
print("\n=== 场景分组统计 ===")
for group in group_colors.keys():
    group_count = sum(1 for cat in class_list if cat in scene_categories.get(group, []))
    print(f"{group}: {group_count}个类别")

print(f"\n总类别数: {len(class_list)}")
