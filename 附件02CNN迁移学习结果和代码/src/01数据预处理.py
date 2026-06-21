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
from create_results_webpage import create_results_webpage_simple
import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
# 数据集文件夹路径
dataset_dir = config.Data_path
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'test')
val_path=os.path.join(dataset_dir, 'val')

train_dataset = datasets.ImageFolder(train_path, train_transform)
test_dataset=datasets.ImageFolder(test_path, test_transform)
val_dataset=datasets.ImageFolder(val_path, test_transform)

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


#获取一个样本
sample_image, sample_label = train_dataset[0]
print(f"\n=== 样本信息 ===")
print(f"图像张量形状: {sample_image.shape}")
print(f"数据类型: {sample_image.dtype}")
print(f"标签: {sample_label}")
print(f"标签对应类别: {train_dataset.classes[sample_label]}")


#--------------建立类别映射关系----------------
# 各类别名称
class_names = train_dataset.classes
n_class = len(class_names)
# 映射关系：类别 到 索引号
train_dataset.class_to_idx
# 映射关系：索引号 到 类别
idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
np.save('15_idx_to_labels.npy', idx_to_labels)
np.save('15_labels_to_idx.npy', train_dataset.class_to_idx)

#---------------建立批次 -------------------------
BATCH_SIZE = 32

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                         )
# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                        )
# 验证集的数据加载器
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=0
                       )


#------------导入迁移学习和定义学习的方式(只训练全连阶层）--------
ARCHITECTURE = config.ARCH
model = models.__dict__[ARCHITECTURE](pretrained=True)
print(model)
# 可选模型：'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#          'densenet161', 'wideresnet18'
# 新建的层默认 requires_grad=True
#model.fc = nn.Linear(model.fc.in_features, n_class)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_class)
# 只微调训练最后一层全连接层的参数，其它层冻结
#optimizer = optim.Adam(model.fc.parameters())
optimizer = optim.Adam(model.classifier[6].parameters())
# optimizer = optim.Adam(model.parameters()) #微调所有层
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 训练轮次 Epoch
EPOCHS = 30
# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#-------------------在训练集上训练-------------------

def train_one_batch(images, labels):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # 把所有记录到日志中
    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    log_train['train_precision'] = precision_score(labels, preds, average='macro')
    log_train['train_recall'] = recall_score(labels, preds, average='macro')
    log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train
#---------------测试集评估函数------------------
def evaluate_val(data_loader=val_loader,loader_name='val'):  # 设置默认值
    '''
    在指定数据集上评估，返回分类评估指标日志
    '''
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # 需要添加到设备
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)


    # 判断是什么数据集
    if data_loader is test_loader:
        prefix = 'test'
    else:
        prefix = 'val'
    # 记录到log中
    log_val = {}

    # 计算分类评估指标
    log_val[f'{prefix}_loss'] = np.mean(loss_list)
    log_val[f'{prefix}_accuracy'] = accuracy_score(labels_list, preds_list)
    log_val[f'{prefix}_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_val[f'{prefix}_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_val[f'{prefix}_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_val

#------------------可视化记录-----------------
#API :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
import wandb
# 在初始化前设置环境变量
os.environ['WANDB_DISABLE_JUPYTER_TELEMETRY'] = 'true'
os.environ['WANDB_SILENT'] = 'true'
# 然后初始化
run = wandb.init(
    project='place15',
    name=time.strftime('%m%d%H%M%S')
)
print(f" WandB 初始化成功！Run ID: {run.id}")

#-----------------------定义保存路径------------------------
class_str=str(len(train_dataset.classes))
output_path=f"{class_str}_{ARCHITECTURE}/"
os.makedirs(output_path, exist_ok=True)
checkpoint_path=f"{class_str}_{ARCHITECTURE}/"+'checkpoint'
os.makedirs(checkpoint_path, exist_ok=True)

epoch = 0
batch_idx = 0
df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0

best_val_accuracy = 0
df_val_log = pd.DataFrame()  # 新增验证集日志
log_val = {}
log_val['epoch'] = 0
images, labels = next(iter(train_loader))


for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}/{EPOCHS}')
    ## 训练阶段
    model.train()
    for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        df_train_log = df_train_log._append(log_train, ignore_index=True)
        wandb.log(log_train)

    lr_scheduler.step()

    ## 验证阶段
    model.eval()
    log_val = evaluate_val(data_loader=val_loader,loader_name='val') # 使用验证集评估
    df_val_log = df_val_log._append(log_val, ignore_index=True)
    wandb.log(log_val)

    # 保存最新的最佳模型文件
    if log_val['val_accuracy'] > best_val_accuracy:
        # 删除旧的最佳模型文件(如有)
        old_best_checkpoint_path = output_path+'checkpoint/best-{:.3f}.pth'.format(best_val_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        best_val_accuracy = log_val['val_accuracy']
        new_best_checkpoint_path = output_path+'checkpoint/best-{:.3f}.pth'.format(log_val['val_accuracy'])
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', 'checkpoint/best-{:.3f}.pth'.format(best_val_accuracy))
        # best_val_accuracy = log_val['test_accuracy']

df_train_log.to_csv(output_path+'训练日志-训练集.csv', index=False)
df_val_log.to_csv(output_path+'训练日志-评估集.csv', index=False)

#----------------载入评估（1）总体评估------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
model = torch.load(new_best_checkpoint_path, weights_only=False)
model = model.eval().to(device)
print(evaluate_val(data_loader=test_loader,loader_name='test'))


#------------------查看每一个的测试结果-----------------------------------
test_dataset.imgs[:10]
img_paths = [each[0] for each in test_dataset.imgs]
df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = test_dataset.targets
df['标注类别名称'] = [idx_to_labels[ID] for ID in test_dataset.targets]
# 记录 top-n 预测结果
n = 3
df_pred = pd.DataFrame()
classes = list(idx_to_labels.values())
for idx, row in tqdm(df.iterrows()):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    pred_dict = {}
    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict['top-{}-预测ID'.format(i)] = pred_ids[i - 1]
        pred_dict['top-{}-预测名称'.format(i)] = idx_to_labels[pred_ids[i - 1]]
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()
    df_pred = df_pred._append(pred_dict, ignore_index=True)
df_pred
df = pd.concat([df, df_pred], axis=1)
df.to_csv(output_path+'测试集预测结果.csv', index=False)

sum(df['标注类别名称'] == df['top-1-预测名称']) / len(df) #top1准确率-0.886
sum(df['top-n预测正确']) / len(df) #就是前面N个有没有预测对的  #topn准确率-0.993
'''
查找预测错误的
true_A = '' #输入具体的label
pred_B = '' #输入具体的label
wrong_df = df[(df['标注类别名称']==true_A)&(df['top-1-预测名称']==pred_B)]

'''

#-------------评估指标（需要用到上面的详细预测结果）df——-------------
'''
macro avg 宏平均：直接将每一类的评估指标求和取平均（算数平均值）
weighted avg 加权平均：按样本数量（support）加权计算评估指标的平均值
'''
from sklearn.metrics import classification_report
#1、准确率
sum(df['标注类别名称'] == df['top-1-预测名称']) / len(df)
#2、top-n准确率
sum(df['top-n预测正确']) / len(df) #就是前面N个有没有预测对的
#3、每一类的结果
report = classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()
accuracy_list = []
for fruit in tqdm(classes):
    df_temp = df[df['标注类别名称']==fruit]
    accuracy = sum(df_temp['标注类别名称'] == df_temp['top-1-预测名称']) / len(df_temp)
    accuracy_list.append(accuracy)
# 计算 宏平均准确率 和 加权平均准确率
acc_macro = np.mean(accuracy_list)
acc_weighted = sum(accuracy_list * df_report.iloc[:-2]['support'] / len(df))
accuracy_list.append(acc_macro)
accuracy_list.append(acc_weighted)
df_report['accuracy'] = accuracy_list
df_report.to_csv(output_path+'各类别准确率评估指标.csv', index_label='类别')

#---------------评估画图 混淆矩阵 （需要用到测试集预测结果df)----------------------
from sklearn.metrics import confusion_matrix
confusion_matrix_model = confusion_matrix(df['标注类别名称'], df['top-1-预测名称'])
confusion_matrix_model.shape
import itertools
def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    """
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    """
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(len(classes))

    plt.title('混淆矩阵', fontsize=30)
    plt.xlabel('预测类别', fontsize=25, c='r')
    plt.ylabel('真实类别', fontsize=25, c='r')
    plt.tick_params(labelsize=16)  # 设置类别文字大小
    plt.xticks(tick_marks, classes, rotation=90)  # 横轴文字旋转
    plt.yticks(tick_marks, classes)

    # 写数字
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=12)

    plt.tight_layout()

    plt.savefig(output_path+'混淆矩阵.pdf', dpi=300)  # 保存图像
    plt.show()
cnf_matrix_plotter(confusion_matrix_model, classes, cmap=plt.cm.Blues)


#-----------评估画图曲线PR-------------
'''
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)
'''
from matplotlib import colors as mcolors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import random
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']
def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg
#查看线的参数
get_line_arg()
plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
# plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.rcParams['font.size'] = 22
plt.grid(True)

ap_list = []
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df['{}-预测置信度'.format(each_class)])
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    AP = average_precision_score(y_test, y_score, average='weighted')
    plt.plot(recall, precision, **get_line_arg(), label=each_class)
    plt.legend()
    ap_list.append(AP)

plt.legend(loc='best', fontsize=12)
plt.savefig(output_path+'各类别PR曲线.pdf', dpi=120, bbox_inches='tight')
plt.show()

#PR 曲线的值被计算为AP，接近1为好的
# 计算 AUC值 的 宏平均 和 加权平均（按照数量加权）
df_report = pd.read_csv(output_path+'各类别准确率评估指标.csv')
macro_avg_auc = np.mean(ap_list)
weighted_avg_auc = sum(ap_list * df_report.iloc[:-2]['support'] / len(df))
ap_list.append(macro_avg_auc)
ap_list.append(weighted_avg_auc)
df_report['AP'] = ap_list
df_report.to_csv(output_path+'各类别准确率评估指标.csv', index=False)

#------------------评估画图指标ROC----------------------------
from sklearn.metrics import roc_curve, auc
df = pd.read_csv(output_path+'测试集预测结果.csv')
plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.rcParams['font.size'] = 22
plt.grid(True)

auc_list = []
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df['{}-预测置信度'.format(each_class)])
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, **get_line_arg(), label=each_class)
    plt.legend()
    auc_list.append(auc(fpr, tpr))

auc_list
plt.legend(loc='best', fontsize=12)
plt.savefig(output_path+'各类别ROC曲线.pdf', dpi=120, bbox_inches='tight')
plt.show()

#记录AUC的结果到表格中
df_report = pd.read_csv(output_path+'各类别准确率评估指标.csv')
# 计算 AUC值 的 宏平均 和 加权平均
macro_avg_auc = np.mean(auc_list)
weighted_avg_auc = sum(auc_list * df_report.iloc[:-2]['support'] / len(df))
auc_list.append(macro_avg_auc)
auc_list.append(weighted_avg_auc)
df_report['AUC'] = auc_list
df_report.to_csv(output_path+'各类别准确率评估指标.csv', index=False)


#------创建create 可视化结果----------
# 准备训练数据
train_image_paths = [each[0] for each in train_dataset.imgs]
train_labels = [idx_to_labels[ID] for ID in train_dataset.targets]
# 获取类别信息
categories = list(idx_to_labels.values())
# 创建缩略类别名（如果没有，用前3个字符）
abbr_categories = [cat[:3] for cat in categories]
# 创建结果网页
create_results_webpage_simple(
    df=df,
    train_image_paths=train_image_paths,
    train_labels=train_labels,
    categories=categories,
    abbr_categories=abbr_categories,
    output_dir=f"{class_str}_{ARCHITECTURE}_results_webpage"
)
print("结果网页已创建在 results_webpage 目录中")
