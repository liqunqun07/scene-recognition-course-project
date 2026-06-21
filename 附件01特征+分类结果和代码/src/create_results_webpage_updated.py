"""
创建结果网页
生成可视化的分类结果网页，包含混淆矩阵和示例图像
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前
import matplotlib.pyplot as plt
import numpy as np
import skimage
import glob
import os
from skimage import io
from skimage.transform import resize
from skimage.color import rgba2rgb, gray2rgb

warnings.filterwarnings('ignore', '', UserWarning)


def create_results_webpage(train_image_paths, test_image_paths,
                           train_labels, test_labels,
                           categories, abbr_categories, predicted_categories,
                           output_dir='results_webpage'):
    """
    创建结果网页
    
    参数:
        train_image_paths: 训练图像路径列表
        test_image_paths: 测试图像路径列表
        train_labels: 训练标签列表
        test_labels: 测试标签列表
        categories: 类别列表
        abbr_categories: 缩写类别列表
        predicted_categories: 预测类别列表
        output_dir: 输出目录
    """
    
    print('=' * 80)
    print(f'创建结果网页: {output_dir}/index.html')
    print('=' * 80)
    
    # 参数设置
    num_samples = 2  # 每类显示的样本数
    thumbnail_height = 75  # 缩略图高度（像素）
    num_categories = len(categories)
    
    # 转换为numpy数组
    categories = np.array(categories)
    predicted_categories = np.array(predicted_categories)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    thumbnails_dir = os.path.join(output_dir, 'thumbnails')
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    # 清理旧缩略图
    old_thumbnails = glob.glob(os.path.join(thumbnails_dir, '*.jpg'))
    for f in old_thumbnails:
        try:
            os.remove(f)
        except:
            pass
    
    # 创建混淆矩阵
    confusion_matrix = create_confusion_matrix(test_labels, predicted_categories, 
                                              categories, num_categories)
    
    # 保存混淆矩阵图
    save_confusion_matrix(confusion_matrix, categories, abbr_categories, 
                         num_categories, output_dir)
    
    # 计算准确率
    accuracy = np.mean(np.diag(confusion_matrix))
    print(f'准确率 (混淆矩阵对角线均值): {accuracy:.3%}')
    
    # 创建HTML文件
    html_path = os.path.join(output_dir, 'index.html')
    
    with open(html_path, 'w+', encoding='utf-8') as f:
        # 写入HTML头部
        write_html_header(f)
        
        # 写入标题和混淆矩阵
        f.write('<div class="container">\n\n')
        f.write('<center>\n')
        f.write('<h1>场景分类结果可视化</h1>\n')
        f.write('<img src="confusion_matrix.png">\n\n')
        f.write('<br>\n')
        f.write(f'准确率 (混淆矩阵对角线均值) 为 {accuracy:.3f}\n')
        f.write('<p>\n\n')
        
        # 创建结果表格
        write_results_table_header(f, num_samples)
        
        # 为每个类别创建行
        for i, cat in enumerate(categories):
            write_category_row(f, i, cat, train_image_paths, test_image_paths,
                             train_labels, test_labels, predicted_categories,
                             categories, confusion_matrix, num_samples, 
                             thumbnail_height, thumbnails_dir)
        
        # 写入表格底部
        write_results_table_footer(f, num_samples)
        
        # 写入HTML尾部
        write_html_footer(f)
    
    print(f'结果网页已保存到: {html_path}')
    print('=' * 80)


def create_confusion_matrix(test_labels, predicted_categories, categories, num_categories):
    """创建混淆矩阵"""
    confusion_matrix = np.zeros((num_categories, num_categories))
    
    for i, cat in enumerate(predicted_categories):
        try:
            row = np.argwhere(categories == test_labels[i])[0][0]
            column = np.argwhere(categories == predicted_categories[i])[0][0]
            confusion_matrix[row][column] += 1
        except:
            print(f"警告: 处理预测 {i} 时出错")
            continue
    
    # 标准化
    num_test_per_cat = len(test_labels) / num_categories
    confusion_matrix = confusion_matrix / float(num_test_per_cat)
    
    return confusion_matrix


def save_confusion_matrix(confusion_matrix, categories, abbr_categories, 
                         num_categories, output_dir):
    """保存混淆矩阵图"""
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, cmap='plasma', interpolation='nearest')
    
    # 设置刻度
    plt.xticks(np.arange(num_categories), abbr_categories, rotation=45)
    plt.yticks(np.arange(num_categories), categories)
    
    plt.colorbar()
    plt.tight_layout()
    
    matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(matrix_path, bbox_inches='tight', dpi=100)
    plt.close()


def write_category_row(f, i, cat, train_image_paths, test_image_paths,
                      train_labels, test_labels, predicted_categories,
                      categories, confusion_matrix, num_samples, 
                      thumbnail_height, thumbnails_dir):
    """为单个类别写入表格行"""
    f.write('<tr>\n')
    
    # 类别名称
    f.write('<td>')
    f.write(f'{cat}')
    f.write('</td>\n')
    
    # 类别准确率
    f.write('<td>')
    f.write(f'{confusion_matrix[i][i]:.3f}')
    f.write('</td>\n')
    
    # 收集样本
    train_examples = np.take(train_image_paths, np.argwhere(train_labels == cat))
    true_positives = np.take(test_image_paths,
                             np.argwhere(np.logical_and(test_labels == cat, 
                                                       predicted_categories == cat)))
    
    false_positive_inds = np.argwhere(
        np.logical_and(np.invert(cat == test_labels), cat == predicted_categories))
    false_positives = np.take(test_image_paths, false_positive_inds)
    false_positive_labels = np.take(test_labels, false_positive_inds)
    
    false_negative_inds = np.argwhere(
        np.logical_and(cat == test_labels, np.invert(cat == predicted_categories)))
    false_negatives = np.take(test_image_paths, false_negative_inds)
    false_negative_labels = np.take(predicted_categories, false_negative_inds)
    
    # 随机打乱
    np.random.shuffle(train_examples)
    np.random.shuffle(true_positives)
    
    rng_state = np.random.get_state()
    np.random.shuffle(false_positives)
    np.random.set_state(rng_state)
    np.random.shuffle(false_positive_labels)
    
    rng_state = np.random.get_state()
    np.random.shuffle(false_negatives)
    np.random.set_state(rng_state)
    np.random.shuffle(false_negative_labels)
    
    # 写入训练样本
    write_sample_images(f, train_examples, cat, num_samples, thumbnail_height, 
                       thumbnails_dir, 'LightBlue')
    
    # 写入真正例
    write_sample_images(f, true_positives, cat, num_samples, thumbnail_height, 
                       thumbnails_dir, 'LightGreen')
    
    # 写入假正例（带标签）
    write_sample_images_with_labels(f, false_positives, false_positive_labels, 
                                   cat, num_samples, thumbnail_height, 
                                   thumbnails_dir, 'LightCoral')
    
    # 写入假负例（带标签）
    write_sample_images_with_labels(f, false_negatives, false_negative_labels, 
                                   cat, num_samples, thumbnail_height, 
                                   thumbnails_dir, '#FFBB55')
    
    f.write('</tr>\n')


def write_sample_images(f, images, cat, num_samples, thumbnail_height, 
                       thumbnails_dir, bgcolor):
    """写入样本图像（无标签）"""
    for j in range(num_samples):
        if j < len(images):
            try:
                img_path = images[j][0] if len(images[j]) > 0 else images[j]
                img_path = str(img_path)
                
                # 处理图像
                tmp = safe_read_image(img_path)
                if tmp is None:
                    f.write(f'<td bgcolor={bgcolor}></td>\n')
                    continue
                
                tmp = process_image_dimensions(tmp)
                height, width = rescale(tmp.shape, thumbnail_height)
                tmp = resize_image_for_display(tmp, (height, width))
                
                # 保存缩略图
                name = os.path.basename(img_path)
                thumbnail_path = os.path.join(thumbnails_dir, f'{cat}_{name}')
                save_thumbnail(tmp, thumbnail_path)
                
                # 写入HTML
                rel_path = os.path.relpath(thumbnail_path, os.path.dirname(thumbnails_dir))
                f.write(f'<td bgcolor={bgcolor}>')
                f.write(f'<img src="{rel_path}" width={width} height={height}>')
                f.write('</td>\n')
            except Exception as e:
                print(f"处理图像失败: {e}")
                f.write(f'<td bgcolor={bgcolor}></td>\n')
        else:
            f.write(f'<td bgcolor={bgcolor}></td>\n')


def write_sample_images_with_labels(f, images, labels, cat, num_samples, 
                                   thumbnail_height, thumbnails_dir, bgcolor):
    """写入样本图像（带标签）"""
    for j in range(num_samples):
        if j < len(images):
            try:
                img_path = images[j][0] if len(images[j]) > 0 else images[j]
                img_path = str(img_path)
                label = labels[j][0] if len(labels[j]) > 0 else labels[j]
                label = str(label)
                
                # 处理图像
                tmp = safe_read_image(img_path)
                if tmp is None:
                    f.write(f'<td bgcolor={bgcolor}></td>\n')
                    continue
                
                tmp = process_image_dimensions(tmp)
                height, width = rescale(tmp.shape, thumbnail_height)
                tmp = resize_image_for_display(tmp, (height, width))
                
                # 保存缩略图
                name = os.path.basename(img_path)
                thumbnail_path = os.path.join(thumbnails_dir, f'{cat}_{name}')
                save_thumbnail(tmp, thumbnail_path)
                
                # 写入HTML
                rel_path = os.path.relpath(thumbnail_path, os.path.dirname(thumbnails_dir))
                f.write(f'<td bgcolor={bgcolor}>')
                f.write(f'<img src="{rel_path}" width={width} height={height}>')
                f.write(f'<br><small>{label}</small>')
                f.write('</td>\n')
            except Exception as e:
                print(f"处理图像失败: {e}")
                f.write(f'<td bgcolor={bgcolor}></td>\n')
        else:
            f.write(f'<td bgcolor={bgcolor}></td>\n')


def safe_read_image(image_path):
    """安全地读取图像"""
    try:
        return io.imread(image_path)
    except Exception as e:
        print(f"读取图像失败 {image_path}: {e}")
        return None


def process_image_dimensions(image):
    """处理图像维度"""
    # 处理4D图像
    if len(image.shape) == 4:
        image = image[0]
    
    # 处理RGBA
    if len(image.shape) == 3 and image.shape[2] == 4:
        try:
            image = rgba2rgb(image)
        except:
            image = image[:, :, :3]
    
    # 处理单通道彩色图像
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    
    # 转换灰度图为RGB
    if len(image.shape) == 2:
        try:
            image = gray2rgb(image)
        except:
            image = np.stack([image] * 3, axis=-1)
    
    return image


def resize_image_for_display(image, output_shape):
    """调整图像大小用于显示"""
    height, width = output_shape
    
    try:
        if len(image.shape) == 2:
            return resize(image, (height, width), anti_aliasing=True)
        elif len(image.shape) == 3:
            return resize(image, (height, width), anti_aliasing=True, preserve_range=True)
        else:
            return np.zeros((height, width, 3), dtype=np.float32)
    except Exception as e:
        print(f"调整图像大小失败: {e}")
        return np.zeros((height, width, 3), dtype=np.float32)


def rescale(dims, thumbnail_height):
    """计算缩略图尺寸"""
    if len(dims) < 2:
        return (thumbnail_height, thumbnail_height)
    
    if len(dims) == 2:
        original_height, original_width = dims[0], dims[1]
    elif len(dims) == 3:
        original_height, original_width = dims[0], dims[1]
    elif len(dims) == 4:
        original_height, original_width = dims[1], dims[2]
    else:
        return (thumbnail_height, thumbnail_height)
    
    scale_factor = thumbnail_height / original_height
    new_height = thumbnail_height
    new_width = int(round(original_width * scale_factor))
    
    return (new_height, new_width)


def save_thumbnail(image, output_path):
    """保存缩略图"""
    # 确保是uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    try:
        io.imsave(output_path, image, check_contrast=False)
    except Exception as e:
        print(f"保存缩略图失败 {output_path}: {e}")


def write_html_header(f):
    """写入HTML头部"""
    f.write('<!DOCTYPE html>\n')
    f.write('<html>\n')
    f.write('<head>\n')
    f.write('<meta charset="UTF-8">\n')
    f.write('<link href="http://fonts.googleapis.com/css?family=Nunito:300|Crimson+Text|Droid+Sans+Mono" rel="stylesheet" type="text/css">\n')
    f.write('<style type="text/css">\n')
    f.write('body { margin: 0px; width: 100%; font-family: "Crimson Text", serif; background: #fcfcfc; }\n')
    f.write('table td { text-align: center; vertical-align: middle; }\n')
    f.write('h1 { font-family: "Nunito", sans-serif; font-weight: normal; font-size: 28px; margin: 25px 0px 0px 0px; text-transform: lowercase; }\n')
    f.write('.container { margin: 0px auto 0px auto; width: 1160px; }\n')
    f.write('</style>\n')
    f.write('</head>\n')
    f.write('<body>\n\n')


def write_results_table_header(f, num_samples):
    """写入结果表格头部"""
    f.write('<table border=0 cellpadding=4 cellspacing=1>\n')
    f.write('<tr>\n')
    f.write('<th>类别名称</th>\n')
    f.write('<th>准确率</th>\n')
    f.write(f'<th colspan={num_samples}>训练样本</th>\n')
    f.write(f'<th colspan={num_samples}>真正例</th>\n')
    f.write(f'<th colspan={num_samples}>假正例（真实标签）</th>\n')
    f.write(f'<th colspan={num_samples}>假负例（错误预测标签）</th>\n')
    f.write('</tr>\n')


def write_results_table_footer(f, num_samples):
    """写入结果表格底部"""
    f.write('<tr>\n')
    f.write('<th>类别名称</th>\n')
    f.write('<th>准确率</th>\n')
    f.write(f'<th colspan={num_samples}>训练样本</th>\n')
    f.write(f'<th colspan={num_samples}>真正例</th>\n')
    f.write(f'<th colspan={num_samples}>假正例（真实标签）</th>\n')
    f.write(f'<th colspan={num_samples}>假负例（错误预测标签）</th>\n')
    f.write('</tr>\n')
    f.write('</table>\n')


def write_html_footer(f):
    """写入HTML尾部"""
    f.write('</center>\n\n\n')
    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')
