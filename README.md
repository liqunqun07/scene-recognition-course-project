# Scene Recognition Course Project

本仓库整理了计算机视觉课程项目的代码、实验报告和部分结果文件。项目围绕场景识别任务，比较了传统特征方法、CNN 迁移学习方法和 CLIP 微调方法，并额外包含旅游场景数据集的爬取与整理脚本。

## 项目内容

- `report-final-project.pdf`：项目报告，可公开阅读。
- `附件01特征+分类结果和代码/`：传统场景识别方法，包括 Tiny Image、Bag of Words、CNN 特征与 NN/SVM 分类器。
- `附件02CNN迁移学习结果和代码/`：基于 torchvision 预训练模型的迁移学习实验，包括 ResNet、VGG 等模型。
- `附件03CLIP_结果和代码/`：基于 Hugging Face CLIP 模型的微调、预测和语义特征可视化代码。
- `附件旅游数据集爬虫代码/`：旅游场景图片爬取、合并、统计、清洗和划分脚本。
- `附件所有测试集RESNET和CLIP预测结果/`：保留的部分测试集预测结果 CSV。
- `附件57类_语义特征UMAP二维降维_交互式分组.html`：CLIP 语义特征 UMAP 可视化结果。

完整原始数据集、模型权重、`.npy` 中间结果和大量网页缩略图未放入仓库。仓库保留了全部代码、报告、汇总预测 CSV、训练/验证日志、指标文件、混淆矩阵和 PR/ROC 图等可复现实验结论所需的代表性结果。

## 环境安装

建议使用 Python 3.10 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要运行爬虫，还需要安装 Playwright 浏览器：

```bash
playwright install
```

PyTorch 的安装方式和本机 CUDA/MPS 环境有关。如默认 `pip install -r requirements.txt` 不适配 GPU，请按 PyTorch 官网说明重新安装 `torch` 和 `torchvision`。

## 数据准备

代码默认不包含完整数据集。请按以下结构准备数据：

```text
data/
  15scene/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  67indoor/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  tourism_pic/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg
```

传统方法的数据路径在 `附件01特征+分类结果和代码/src/config.yaml` 中配置。CNN 和 CLIP 脚本也支持环境变量覆盖：

```bash
export SCENE_DATA_PATH=/path/to/15scene-or-67indoor
export TOURISM_DATA_PATH=/path/to/tourism_pic
```

## 运行传统特征/分类实验

进入传统方法代码目录：

```bash
cd "附件01特征+分类结果和代码/src"
```

查看可用数据集：

```bash
python main_improved.py --list-datasets
```

运行单个实验：

```bash
python main_improved.py --dataset 15scene --feature tiny_image --classifier nearest_neighbor
python main_improved.py --dataset 15scene --feature bag_of_words --classifier support_vector_machine
python main_improved.py --dataset 67indoor --feature cnn --cnn-model resnet18 --classifier support_vector_machine
```

运行预设组合：

```bash
python main_improved.py --dataset 15scene --run-all
python main_improved.py --dataset 67indoor --run-all
```

## 运行 CNN 迁移学习实验

进入迁移学习代码目录：

```bash
cd "附件02CNN迁移学习结果和代码/src"
```

配置数据路径和模型：

```bash
export SCENE_DATA_PATH=/path/to/15scene
export SCENE_CNN_ARCH=resnet18
```

如需要从测试集划分验证集：

```bash
python 00新增数据划分.py
```

训练、评估并导出结果：

```bash
python 01数据预处理.py
```

`config.py` 中可调整模型结构，支持 `resnet18`、`resnet34`、`resnet50`、`vgg16` 等 torchvision 预训练模型。

## 运行 CLIP 微调与语义特征可视化

进入 CLIP 代码目录：

```bash
cd "附件03CLIP_结果和代码/src_clip0208"
```

配置数据路径：

```bash
export TOURISM_DATA_PATH=/path/to/tourism_pic
```

训练和评估 CLIP：

```bash
python 01数据预处理_CLIP.py
```

生成测试集语义特征并进行 UMAP 可视化：

```bash
export CLIP_BEST_MODEL_PATH=/path/to/best_model.pth
export CLIP_RESULT_PATH=/path/to/result_dir
python 语义特征-颜色区分.py
```

默认 CLIP 模型为 `openai/clip-vit-base-patch16`，训练策略为 `only_projection`。可在 `config_clip.py` 中修改 batch size、学习率、训练轮数、CLIP 模型和微调策略。

## 旅游数据集脚本

进入爬虫脚本目录：

```bash
cd "附件旅游数据集爬虫代码"
```

爬取 Google 图片：

```bash
python 谷歌爬虫0214.py
```

统计数据分布：

```bash
TOURISM_DATA_PATH=/path/to/tourism_pic python 数据统计.py
```

合并图片文件夹前建议先使用演习模式：

```bash
python 图片合并.py /path/to/folder_a /path/to/folder_b --dry-run
```

## 结果说明

仓库中保留了部分代表性实验输出：

- 传统方法结果：`results_base_15/` 和 `results_base_67/` 中的 `results.txt`、`config.json`、`confusion_matrix.png` 等。
- CNN 迁移学习结果：`scene15/`、`indoor67/` 中的训练日志、测试集预测 CSV、指标 CSV、混淆矩阵、PR/ROC 曲线。
- CLIP 结果：`CLIP结果/` 中的训练日志、验证日志、测试集预测 CSV、混淆矩阵、PR/ROC 曲线。
- 汇总预测结果：`附件所有测试集RESNET和CLIP预测结果/`。

未上传内容：

- 完整原始图像数据集。
- 模型权重和 checkpoint。
- `.npy` 特征/预测中间文件。
- 结果网页中的大量 `thumbnails/` 缩略图。

这些文件可通过重新运行代码生成，或根据本地实验目录补充。
