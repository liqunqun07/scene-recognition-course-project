import os
target_dir = os.environ.get("CNN_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.chdir(target_dir)
Data_path = os.environ.get("SCENE_DATA_PATH", os.path.join(target_dir, "data", "15scene"))
ARCH = os.environ.get("SCENE_CNN_ARCH", "vgg16")
# 可选模型：'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#          'densenet161', 'wideresnet18'
# 模型是从 PyTorch 的 torchvision models 迁移过来，model = models.__dict__[ARCHITECTURE](pretrained=True)
# 第三步：准备训练集和测试集
