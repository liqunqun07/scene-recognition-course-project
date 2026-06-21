import cv2 as cv
import matplotlib
import os
import warnings
warnings.filterwarnings("ignore")
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

# ============ CLIP特定导入 ============
from transformers import CLIPModel, CLIPProcessor

print(f"当前工作目录: {project_root}")
target_dir = os.environ.get("CLIP_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.chdir(target_dir)
from create_results_webpage import create_results_webpage_simple
import config_clip as config

# 设备配置
if config.DEVICE == "auto":
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
else:
    device = torch.device(config.DEVICE)
print(f"使用设备: {device}")



class CLIPImageFolder(Dataset):
    """
    CLIP版本的ImageFolder，保持与原始ImageFolder相同的接口
    过滤掉非目录和非图像文件
    """

    def __init__(self, root_dir, processor, is_train=True):
        self.root_dir = root_dir
        self.processor = processor
        self.is_train = is_train

        # 获取类别信息 - 过滤掉隐藏文件和非目录
        self.classes = sorted([
            cls for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
               and not cls.startswith('.')  # 过滤掉以点开头的隐藏文件
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 加载所有图像路径和标签
        self.imgs = []
        self.targets = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                # 过滤隐藏文件和检查图像后缀
                if (not img_name.startswith('.')) and img_name.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_dir, img_name)
                    self.imgs.append((img_path, self.class_to_idx[class_name]))
                    self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label
# ============ CLIP函数 ============
def clip_collate_fn(batch, processor, class_names):
    """
    CLIP专用的batch整理函数
    将图像和对应的类别名称一起处理
    """
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    # 获取对应的类别名称作为文本
    texts = [class_names[label.item()] for label in labels]
    
    # 使用CLIP processor处理
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    
    return {
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

# ============ 数据集路径配置============
dataset_dir = config.Data_path
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'test')
val_path = os.path.join(dataset_dir, 'val')

# ============ 加载CLIP模型和处理器 ============
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

# ============ 创建CLIP数据集============
train_dataset = CLIPImageFolder(train_path, processor, is_train=True)
test_dataset = CLIPImageFolder(test_path, processor, is_train=False)
val_dataset = CLIPImageFolder(val_path, processor, is_train=False)

print('训练集路径', train_path)
print('测试集路径', test_path)
print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)
print('验证集图像数量', len(val_dataset))
print('类别个数', len(val_dataset.classes))

# 获取一个样本
sample_image, sample_label = train_dataset[0]
print(f"\n=== 样本信息 ===")
print(f"图像类型: {type(sample_image)}")
print(f"标签: {sample_label}")
print(f"标签对应类别: {train_dataset.classes[sample_label]}")

# ============ 建立类别映射关系============
class_names = train_dataset.classes
n_class = len(class_names)
train_dataset.class_to_idx
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
np.save('{}_idx_to_labels.npy'.format(n_class), idx_to_labels)
np.save('{}_labels_to_idx.npy'.format(n_class), train_dataset.class_to_idx)

# ============ 建立批次============
print("\n" + "="*60)
print("🔍 数据集诊断")
print("="*60)
print(f"类别总数: {len(train_dataset.classes)}")
print(f"类别列表: {train_dataset.classes}")
print(f"标签范围: {min(train_dataset.targets)} ~ {max(train_dataset.targets)}")
print("="*60 + "\n")

BATCH_SIZE = config.BATCH_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = config.NUM_WORKERS

# 创建包装的collate函数
def wrapped_collate_fn(batch):
    return clip_collate_fn(batch, processor, class_names)

# 训练集的数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=wrapped_collate_fn
)

# 测试集的数据加载器
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=wrapped_collate_fn
)

# 验证集的数据加载器
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=wrapped_collate_fn
)

# ============ CLIP模型配置============
model = model.to(device)
print(model)

# CLIP微调策略配置
print(f"\n训练策略: {config.TRAINING_STRATEGY}")

if config.TRAINING_STRATEGY == "freeze_vision":
    # 冻结视觉编码器
    print("冻结视觉编码器，只训练文本编码器")
    for param in model.vision_model.parameters():
        param.requires_grad = False
    # 使用AdamW优化器（参考代码中使用的）
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=config.LEARNING_RATE)
    
elif config.TRAINING_STRATEGY == "freeze_text":
    # 冻结文本编码器
    print("冻结文本编码器，只训练视觉编码器")
    for param in model.text_model.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=config.LEARNING_RATE)
    
elif config.TRAINING_STRATEGY == "last_layers":
    # 只微调最后几层
    print(f"只微调最后 {config.NUM_LAYERS_TO_FINETUNE} 层")
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻最后几层
    num_layers = config.NUM_LAYERS_TO_FINETUNE
    for layer in list(model.vision_model.encoder.layers)[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=config.LEARNING_RATE)

elif config.TRAINING_STRATEGY == "only_projection":  # 新增这里
    # 只训练投影层
    print("只训练投影层 (visual_projection 和 text_projection)")
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻投影层
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    for param in model.text_projection.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=config.LEARNING_RATE)
    
else:  # full_finetune
    print("全部参数微调")
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 打印可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

EPOCHS = config.EPOCHS
lr_scheduler2 = lr_scheduler.StepLR(optimizer,
                                    step_size=config.LR_STEP_SIZE, 
                                    gamma=config.LR_GAMMA)

# ============ 训练函数============
def train_one_batch(batch):
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_loss=True
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # 只记录loss
    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    log_train['train_loss'] = loss.detach().cpu().numpy()

    return log_train

# ============ 验证函数============
def evaluate_val(data_loader=val_loader, loader_name='val'):
    '''
    在指定数据集上评估，返回分类评估指标日志
    CLIP版本：使用所有类别进行预测
    '''
    model.eval()
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            all_class_texts = [f"{cls}" for cls in class_names]  # 或 f"a photo of a {cls}"
            text_inputs = processor(
                text=all_class_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            image_outputs = model.get_image_features(pixel_values=pixel_values)
            image_features = image_outputs  # 如果是tensor直接用，如果是对象取.pooler_output
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output

            # 计算文本特征
            text_outputs = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            text_features = text_outputs
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output

            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算相似度 [batch_size, num_classes]
            logits_per_image = model.logit_scale.exp() * image_features @ text_features.T

            # 预测类别
            _, preds = torch.max(logits_per_image, 1)  # 现在preds是真正的类别ID

            # 计算训练用的对比损失（仅用于记录loss）
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            loss = outputs.loss

            preds = preds.cpu().numpy()
            loss_value = loss.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            loss_list.append(loss_value)
            labels_list.extend(labels_np)
            preds_list.extend(preds)

    model.train()

    # 判断是什么数据集
    if data_loader is test_loader:
        prefix = 'test'
    else:
        prefix = 'val'

    log_val = {}
    log_val[f'{prefix}_loss'] = np.mean(loss_list)
    log_val[f'{prefix}_accuracy'] = accuracy_score(labels_list, preds_list)
    log_val[f'{prefix}_precision'] = precision_score(labels_list, preds_list, average='macro', zero_division=0)
    log_val[f'{prefix}_recall'] = recall_score(labels_list, preds_list, average='macro', zero_division=0)
    log_val[f'{prefix}_f1-score'] = f1_score(labels_list, preds_list, average='macro', zero_division=0)

    return log_val
# ============ 在训练循环开始前，检查 ============
print("检查第一个batch:")
first_batch = next(iter(train_loader))
print(f"pixel_values shape: {first_batch['pixel_values'].shape}")
print(f"input_ids shape: {first_batch['input_ids'].shape}")
print(f"labels: {first_batch['labels'].numpy()}")
print(f"labels 范围: {first_batch['labels'].min().item()} ~ {first_batch['labels'].max().item()}")
print()

# ============ WandB初始化============
import wandb
os.environ['WANDB_DISABLE_JUPYTER_TELEMETRY'] = 'true'
os.environ['WANDB_SILENT'] = 'true'
run = wandb.init(
    project=config.WANDB_PROJECT,
    name=time.strftime('%m%d%H%M%S'),
    config={
        "model": config.CLIP_MODEL_NAME,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "strategy": config.TRAINING_STRATEGY,
    }
)
print(f"WandB 初始化成功！Run ID: {run.id}")

# ============ 定义保存路径 ============
class_str = str(len(train_dataset.classes))
# 从模型名称提取简短版本
model_short_name = config.CLIP_MODEL_NAME.split('/')[-1].replace('-', '_')
ARCHITECTURE = f"CLIP_{model_short_name}"

if config.OUTPUT_DIR_MODE == "auto":
    output_path = f"{class_str}_{ARCHITECTURE}/"
else:
    output_path = f"{config.CUSTOM_OUTPUT_DIR}/"

os.makedirs(output_path, exist_ok=True)
print(f"输出路径: {output_path}")

# ============ 训练循环============
epoch = 0
batch_idx = 0
best_val_acc = 0

df_train_log = pd.DataFrame()
df_val_log = pd.DataFrame()

for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}/{EPOCHS}')
    
    model.train()
    
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        log_train = train_one_batch(batch)
        df_train_log = df_train_log._append(log_train, ignore_index=True)
        wandb.log(log_train)
    
    lr_scheduler2.step()
    
    log_val = evaluate_val(val_loader, 'val')
    df_val_log = df_val_log._append(log_val, ignore_index=True)
    wandb.log(log_val)
    
    val_acc = log_val['val_accuracy']
    print(f"Epoch {epoch} - Val Accuracy: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(output_path, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'保存最佳模型，验证集准确率: {val_acc:.4f}')
        
        # 同时保存为Transformers格式
        best_transformers_path = os.path.join(output_path, 'best_model_transformers')
        model.save_pretrained(best_transformers_path)
        processor.save_pretrained(best_transformers_path)
        print(f'Transformers格式模型已保存到: {best_transformers_path}')
    
    # 保存每个epoch的checkpoint
    if config.SAVE_EVERY_EPOCH:
        epoch_model_path = os.path.join(output_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f'保存Epoch {epoch}的checkpoint')

# 保存微调后的最终模型到配置指定的路径
final_model_path = config.FINE_TUNED_MODEL_PATH
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
processor.save_pretrained(final_model_path)
print(f"\n✅ 微调完成！模型已保存到: {final_model_path}")

df_train_log.to_csv(output_path + '训练日志.csv', index=False)
df_val_log.to_csv(output_path + '验证日志.csv', index=False)

# ============ 加载最佳模型进行测试 ============
best_model_path = os.path.join(output_path, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path))
model.eval()

# ============ 测试集预测============
img_paths = [each[0] for each in test_dataset.imgs]
df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = test_dataset.targets
df['标注类别名称'] = [idx_to_labels[ID] for ID in test_dataset.targets]

# 记录 top-n 预测结果
n = config.TOP_N
df_pred = pd.DataFrame()
classes = list(idx_to_labels.values())

print("开始测试集预测...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    
    # CLIP预测：需要所有类别名称
    inputs = processor(
        text=classes,
        images=img_pil,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取预测logits
    logits_per_image = outputs.logits_per_image  # [1, n_classes]
    pred_softmax = F.softmax(logits_per_image, dim=1)
    
    pred_dict = {}
    top_n_results = torch.topk(pred_softmax, n)
    pred_ids = top_n_results[1].cpu().detach().numpy().squeeze()
    
    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict[f'top-{i}-预测ID'] = pred_ids[i - 1]
        pred_dict[f'top-{i}-预测名称'] = idx_to_labels[pred_ids[i - 1]]
    
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    
    # 每个类别的预测置信度
    for idx_class, each in enumerate(classes):
        pred_dict[f'{each}-预测置信度'] = pred_softmax[0][idx_class].cpu().detach().numpy()
    
    df_pred = df_pred._append(pred_dict, ignore_index=True)

df = pd.concat([df, df_pred], axis=1)
df.to_csv(output_path + '测试集预测结果.csv', index=False)

print(f"Top-1准确率: {sum(df['标注类别名称'] == df['top-1-预测名称']) / len(df):.4f}")
print(f"Top-{n}准确率: {sum(df['top-n预测正确']) / len(df):.4f}")

# ============ 评估指标============
from sklearn.metrics import classification_report

report = classification_report(df['标注类别名称'], df['top-1-预测名称'], 
                               target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()

accuracy_list = []
for fruit in tqdm(classes):
    df_temp = df[df['标注类别名称'] == fruit]
    accuracy = sum(df_temp['标注类别名称'] == df_temp['top-1-预测名称']) / len(df_temp)
    accuracy_list.append(accuracy)

acc_macro = np.mean(accuracy_list)
acc_weighted = sum(accuracy_list * df_report.iloc[:-2]['support'] / len(df))
accuracy_list.append(acc_macro)
accuracy_list.append(acc_weighted)
df_report['accuracy'] = accuracy_list
df_report.to_csv(output_path + '各类别准确率评估指标.csv', index_label='类别')

# ============ 混淆矩阵============
from sklearn.metrics import confusion_matrix
import itertools

confusion_matrix_model = confusion_matrix(df['标注类别名称'], df['top-1-预测名称'])

def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.title('混淆矩阵', fontsize=30)
    plt.xlabel('预测类别', fontsize=25, c='r')
    plt.ylabel('真实类别', fontsize=25, c='r')
    plt.tick_params(labelsize=16)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path + '混淆矩阵.pdf', dpi=300)
    plt.show()

cnf_matrix_plotter(confusion_matrix_model, classes, cmap=plt.cm.Blues)

# ============ PR曲线============
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import random

random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
linestyle = ['--', '-.', '-']

def get_line_arg():
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    return line_arg

plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.rcParams['font.size'] = 22
plt.grid(True)

ap_list = []
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df[f'{each_class}-预测置信度'])
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    AP = average_precision_score(y_test, y_score, average='weighted')
    plt.plot(recall, precision, **get_line_arg(), label=each_class)
    ap_list.append(AP)

plt.legend(loc='best', fontsize=12)
plt.savefig(output_path + '各类别PR曲线.pdf', dpi=120, bbox_inches='tight')
plt.show()

df_report = pd.read_csv(output_path + '各类别准确率评估指标.csv')
macro_avg_ap = np.mean(ap_list)
weighted_avg_ap = sum(ap_list * df_report.iloc[:-2]['support'] / len(df))
ap_list.append(macro_avg_ap)
ap_list.append(weighted_avg_ap)
df_report['AP'] = ap_list
df_report.to_csv(output_path + '各类别准确率评估指标.csv', index=False)

# ============ ROC曲线============
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.plot([0, 1], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.rcParams['font.size'] = 22
plt.grid(True)

auc_list = []
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df[f'{each_class}-预测置信度'])
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, **get_line_arg(), label=each_class)
    auc_list.append(auc(fpr, tpr))

plt.legend(loc='best', fontsize=12)
plt.savefig(output_path + '各类别ROC曲线.pdf', dpi=120, bbox_inches='tight')
plt.show()

df_report = pd.read_csv(output_path + '各类别准确率评估指标.csv')
macro_avg_auc = np.mean(auc_list)
weighted_avg_auc = sum(auc_list * df_report.iloc[:-2]['support'] / len(df))
auc_list.append(macro_avg_auc)
auc_list.append(weighted_avg_auc)
df_report['AUC'] = auc_list
df_report.to_csv(output_path + '各类别准确率评估指标.csv', index=False)

# ============ 创建可视化结果============
if config.CREATE_WEBPAGE:
    train_image_paths = [each[0] for each in train_dataset.imgs]
    train_labels = [idx_to_labels[ID] for ID in train_dataset.targets]
    categories = list(idx_to_labels.values())
    abbr_categories = [cat[:3] for cat in categories]

    create_results_webpage_simple(
        df=df,
        train_image_paths=train_image_paths,
        train_labels=train_labels,
        categories=categories,
        abbr_categories=abbr_categories,
        output_dir=f"{class_str}_{ARCHITECTURE}_results_webpage"
    )
    print("结果网页已创建在 results_webpage 目录中")

# ============ 零样本测试============
if config.ENABLE_ZERO_SHOT_TEST:
    print("\n" + "="*60)
    print("开始零样本测试...")
    print("="*60)
    
    # 组合训练类别和新类别
    all_zero_shot_labels = list(set(class_names + config.ZERO_SHOT_LABELS))
    print(f"零样本候选标签: {all_zero_shot_labels}")
    
    # 随机选择一些测试图像
    num_test_samples = min(10, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), num_test_samples, replace=False)
    
    print(f"\n测试 {num_test_samples} 个样本的零样本能力:")
    for idx in test_indices:
        img_path, true_label = test_dataset.imgs[idx]
        img_pil = Image.open(img_path).convert('RGB')

        inputs = processor(
            text=all_zero_shot_labels,
            images=img_pil,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits_per_image[0]
        probs = F.softmax(logits, dim=0)
        top3_idx = torch.topk(probs, 3).indices
        
        print(f"\n图像: {os.path.basename(img_path)}")
        print(f"真实标签: {idx_to_labels[true_label]}")
        print(f"Top-3 预测:")
        for i, idx in enumerate(top3_idx):
            print(f"  {i+1}. {all_zero_shot_labels[idx]}: {probs[idx]:.3f}")

wandb.finish()
print("\n" + "="*60)
print("🎉 训练完成！")
print("="*60)
print(f"📊 最佳验证准确率: {best_val_acc:.4f}")
print(f"📁 结果保存在: {output_path}")
print(f"🤖 微调模型保存在: {config.FINE_TUNED_MODEL_PATH}")
print("="*60)
