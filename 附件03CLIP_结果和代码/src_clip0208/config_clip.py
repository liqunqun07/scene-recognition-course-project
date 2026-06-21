import os

# ============ 目录配置 ============
# 项目根目录
target_dir = os.environ.get("CLIP_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.chdir(target_dir)

# ============ 数据集配置 ============
# 数据集路径（包含train/test/val三个子文件夹）
Data_path = os.environ.get("TOURISM_DATA_PATH", os.path.join(target_dir, "data", "tourism_pic"))



# ============ CLIP模型配置 ============
# 可选的CLIP模型列表：
# - "openai/clip-vit-base-patch32"     (ViT-B/32, 轻量级, 速度快)
# - "openai/clip-vit-base-patch16"     (ViT-B/16, 中等, 精度较高)
# - "openai/clip-vit-large-patch14"    (ViT-L/14, 大模型, 精度最高但慢)
# - "openai/clip-vit-large-patch14-336" (ViT-L/14@336px, 最高精度)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
#CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
model_name=str(CLIP_MODEL_NAME).split('/')[-1]

# 本地模型存储路径（下载后缓存，避免重复下载）
LOCAL_MODEL_PATH = os.path.join(target_dir, "{}model".format(model_name), "clip_model")

# 微调后模型保存路径
FINE_TUNED_MODEL_PATH = os.path.join(target_dir,  "{}model".format(model_name), "fine_tuned_clip")

# ============ 训练超参数配置 ============
# 批次大小（根据显存调整，CLIP模型较大，建议16-32）
BATCH_SIZE = 32

# 学习率（CLIP微调建议使用较小的学习率）
LEARNING_RATE = 1e-5

# 训练轮数
EPOCHS = 30

# 学习率调度器参数
LR_STEP_SIZE = 5      # 每隔多少epoch降低学习率
LR_GAMMA = 0.5        # 学习率衰减系数

# ============ 优化策略配置 ============
# 可选优化策略：
# - "freeze_vision": 冻结视觉编码器，只训练文本编码器
# - "freeze_text": 冻结文本编码器，只训练视觉编码器
# - "full_finetune": 全部参数微调（默认）
# - "last_layers": 只微调最后几层
#-only_projection"：训练投影层

TRAINING_STRATEGY = "only_projection"

# 如果选择last_layers策略，指定微调的层数
NUM_LAYERS_TO_FINETUNE = 1

# ============ 数据增强配置 ============
# 是否使用数据增强（对于CLIP可能不太需要，因为已经有强大的预训练）
USE_AUGMENTATION = False

# ============ 日志和保存配置 ============
# WandB项目名称
WANDB_PROJECT = "tourism_CLIP"

# 输出目录命名方式
# "auto": 自动根据类别数和模型名生成（如 "15_CLIP_ViT_B32"）
# "custom": 使用下面的CUSTOM_OUTPUT_DIR
OUTPUT_DIR_MODE = "auto"
CUSTOM_OUTPUT_DIR = "{}_{}_my_clip_results".format(model_name,WANDB_PROJECT)

# 是否保存每个epoch的checkpoint
SAVE_EVERY_EPOCH = False

# ============ 评估配置 ============
# Top-N准确率的N值
TOP_N = 3

# 是否生成可视化结果网页
CREATE_WEBPAGE = True

# ============ 设备配置 ============
DEVICE = "auto"  # auto会自动选择cuda或cpu

# ============ 高级配置 ============
# 是否使用混合精度训练（可以加速训练并减少显存占用）
USE_MIXED_PRECISION = False

# DataLoader的工作进程数（根据CPU核心数调整）
NUM_WORKERS = 0

# 随机种子（保证可复现性）
RANDOM_SEED = 42

# ============ 零样本测试配置 ============
# 是否在训练后进行零样本测试
ENABLE_ZERO_SHOT_TEST = False

# 零样本测试的候选标签（可以包含训练集之外的类别）
ZERO_SHOT_LABELS = [
    # 训练集类别会自动添加
    # 这里可以添加额外的未见过的类别用于测试零样本能力
    "unknown_object",
    "outdoor_scene",
    "indoor_scene"
]

# ============ 配置验证函数 ============
def validate_config():
    """验证配置是否正确"""
    errors = []
    
    # 检查数据路径是否存在
    if not os.path.exists(Data_path):
        errors.append(f"数据集路径不存在: {Data_path}")
    else:
        # 检查是否包含train/test/val文件夹
        required_dirs = ['train', 'test', 'val']
        for dir_name in required_dirs:
            dir_path = os.path.join(Data_path, dir_name)
            if not os.path.exists(dir_path):
                errors.append(f"缺少必需的文件夹: {dir_path}")
    
    # 检查模型名称是否有效
    valid_models = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336"
    ]
    if CLIP_MODEL_NAME not in valid_models:
        errors.append(f"不支持的模型: {CLIP_MODEL_NAME}. 支持的模型: {valid_models}")
    
    # 检查训练策略是否有效
    valid_strategies = ["freeze_vision", "freeze_text", "full_finetune", "last_layers", "only_projection"]
    if TRAINING_STRATEGY not in valid_strategies:
        errors.append(f"不支持的训练策略: {TRAINING_STRATEGY}. 支持的策略: {valid_strategies}")
    
    # 检查超参数范围
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE必须大于0，当前值: {BATCH_SIZE}")
    if LEARNING_RATE <= 0:
        errors.append(f"LEARNING_RATE必须大于0，当前值: {LEARNING_RATE}")
    if EPOCHS <= 0:
        errors.append(f"EPOCHS必须大于0，当前值: {EPOCHS}")
    
    return errors

# ============ 配置打印函数 ============
def print_config():
    """打印当前配置"""
    print("\n" + "="*60)
    print("CLIP迁移学习配置")
    print("="*60)
    print(f"📁 数据集路径: {Data_path}")
    print(f"🤖 CLIP模型: {CLIP_MODEL_NAME}")
    print(f"💾 本地模型路径: {LOCAL_MODEL_PATH}")
    print(f"📦 微调模型保存路径: {FINE_TUNED_MODEL_PATH}")
    print(f"\n训练参数:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - 训练策略: {TRAINING_STRATEGY}")
    if TRAINING_STRATEGY == "last_layers":
        print(f"  - 微调层数: {NUM_LAYERS_TO_FINETUNE}")
    print(f"\n其他配置:")
    print(f"  - Top-N: {TOP_N}")
    print(f"  - WandB项目: {WANDB_PROJECT}")
    print(f"  - 工作进程数: {NUM_WORKERS}")
    print(f"  - 随机种子: {RANDOM_SEED}")
    print("="*60 + "\n")

# ============ 自动验证 ============
if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("\n⚠️  配置错误:")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("\n✅ 配置验证通过!")
        print_config()
