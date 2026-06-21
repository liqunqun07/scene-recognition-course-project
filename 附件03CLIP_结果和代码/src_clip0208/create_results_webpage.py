
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib

# 使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Songti SC', 'STFangsong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import numpy as np
import skimage
import glob
import os
import pandas as pd
from skimage import io
from skimage.transform import resize

# Skimage gives us some lossy conversion errors that we really don't care about
# so we suppress them
warnings.filterwarnings('ignore', '', UserWarning)


def create_results_webpage_from_df(df, train_image_paths, test_image_paths,
                                   train_labels, test_labels,
                                   categories, abbr_categories):
    '''
    从DataFrame创建结果网页
    df: 包含预测结果的DataFrame，必须包含以下列：
        '图像路径' - 图像文件路径
        '标注类别名称' - 真实标签
        'top-1-预测名称' - 预测标签
    '''

    # 从DataFrame提取预测结果
    predicted_categories = df['top-1-预测名称'].values
    test_image_paths_array = df['图像路径'].values
    test_labels_array = df['标注类别名称'].values

    # 调用原始函数
    create_results_webpage(train_image_paths, test_image_paths_array,
                           train_labels, test_labels_array,
                           categories, abbr_categories, predicted_categories)


def create_results_webpage(train_image_paths, test_image_paths,
                           train_labels, test_labels,
                           categories, abbr_categories, predicted_categories):
    '''

    This function creates a webpage (html and images) visualizing the
    classiffication results. This webpage will contain:
     (1) A confusion matrix plot
     (2) A table with one row per category, with 4 columns - training
         examples, true positives, false positives, and false negatives.
    This webpage is similar to the one created for the SUN database in
    2010: http://people.csail.mit.edu/jxiao/SUN/classification397.html
    '''

    print('Creating results_webpage/index.html, thumbnails, and confusion matrix.')

    # Number of examples of training examples, true positives, false positives,
    # and false negatives. Thus the table will be num_samples * 4 images wide
    # (unless there aren't enough images)
    num_samples = 2
    thumbnail_height = 75  # pixels
    num_categories = len(categories)

    # Convert everything over to numpy arrays
    categories = np.array(categories)
    predicted_categories = np.array(predicted_categories)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Delete the old thumbnails, if there are any
    files = glob.glob('results_webpage/thumbnails/*.jpg')
    for f in files:
        os.remove(f)

    if not os.path.isdir('results_webpage'):
        print('Making results_webpage directory.')
        os.mkdir('results_webpage')
    if not os.path.isdir('results_webpage/thumbnails'):
        print('Making thumbnails directory.')
        os.mkdir('results_webpage/thumbnails')

    ### Create And Save Confusion Matrix ###
    # Based on the predicted category for each test case, we will now build a
    # confusion matrix. Entry (i,j) in this matrix well be the proportion of
    # times a test image of ground truth category i was predicted to be
    # category j. An identity matrix is the ideal case. You should expect
    # roughly 50-95% along the diagonal depending on your features,
    # classifiers, and particular categories. For example, suburb is very easy
    # to recognize.
    with open('results_webpage/index.html', 'w+') as f:

        # Initialize the matrix
        confusion_matrix = np.zeros((num_categories, num_categories))

        # Iterate over predicted results (this is like, several hundred items long)
        for i, cat in enumerate(predicted_categories):
            # Find the row and column corresponding to the label of this entry
            # The row is the ground truth label and the column is the found label
            row = np.argwhere(categories == test_labels[i])[0][0]
            column = np.argwhere(categories == predicted_categories[i])[0][0]

            # Add 1 to the matrix for that row/col
            # This way we build up a histogram from our labeled data
            confusion_matrix[row][column] += 1;

        # If the number of training examples and test cases are not equal, this
        # statement will be invalid!
        # TODO: That's an old comment left over from the matlab code that I don't
        # think still applies
        num_test_per_cat = len(test_labels) / num_categories
        confusion_matrix = confusion_matrix / float(num_test_per_cat)
        accuracy = np.mean(np.diag(confusion_matrix))

        print('Accuracy (mean of diagonal of confusion matrix) is {:2.3%}'.format(accuracy))

        # plasma is the most easily-interpreted color map I've found so far
        plt.imshow(confusion_matrix, cmap='plasma', interpolation='nearest')

        # We put the shortened labels (e.g. "sub" for "suburb") on the x axis
        locs, labels = plt.xticks()
        plt.xticks(np.arange(num_categories), abbr_categories)

        # Full labels go on y
        locs, labels = plt.yticks()
        plt.yticks(np.arange(num_categories), categories)

        # Save the result
        plt.savefig('results_webpage/confusion_matrix.png', bbox_inches='tight')

        ## Create webpage header
        f.write('<!DOCTYPE html>\n');
        f.write('<html>\n');
        f.write('<head>\n');
        f.write(
            '<link href=''http://fonts.googleapis.com/css?family=Nunito:300|Crimson+Text|Droid+Sans+Mono'' rel=''stylesheet'' type=''text/css''>\n');
        f.write('<style type="text/css">\n');

        f.write('body {\n');
        f.write('  margin: 0px;\n');
        f.write('  width: 100%;\n');
        f.write('  font-family: ''Crimson Text'', serif;\n');
        f.write('  background: #fcfcfc;\n');
        f.write('}\n');
        f.write('table td {\n');
        f.write('  text-align: center;\n');
        f.write('  vertical-align: middle;\n');
        f.write('}\n');
        f.write('h1 {\n');
        f.write('  font-family: ''Nunito'', sans-serif;\n');
        f.write('  font-weight: normal;\n');
        f.write('  font-size: 28px;\n');
        f.write('  margin: 25px 0px 0px 0px;\n');
        f.write('  text-transform: lowercase;\n');
        f.write('}\n');
        f.write('.container {\n');
        f.write('  margin: 0px auto 0px auto;\n');
        f.write('  width: 1160px;\n');
        f.write('}\n');

        f.write('</style>\n');
        f.write('</head>\n');
        f.write('<body>\n\n');

        f.write('<div class="container">\n\n\n');
        f.write('<center>\n');
        f.write('<h1>Scene classification results visualization</h1>\n');
        f.write('<img src="confusion_matrix.png">\n\n');
        f.write('<br>\n');
        f.write('Accuracy (mean of diagonal of confusion matrix) is %2.3f\n' % (accuracy));
        f.write('<p>\n\n');

        ## Create results table
        f.write('<table border=0 cellpadding=4 cellspacing=1>\n');
        f.write('<tr>\n');
        f.write('<th>Category name</th>\n');
        f.write('<th>Accuracy</th>\n');
        f.write('<th colspan=%d>Sample training images</th>\n' % num_samples);
        f.write('<th colspan=%d>Sample true positives</th>\n' % num_samples);
        f.write('<th colspan=%d>False positives with true label</th>\n' % num_samples);
        f.write('<th colspan=%d>False negatives with wrong predicted label</th>\n' % num_samples);
        f.write('</tr>\n');

        for i, cat in enumerate(categories):
            f.write('<tr>\n');

            f.write('<td>');  # category name
            f.write('%s' % cat);
            f.write('</td>\n');

            f.write('<td>');  # category accuracy
            f.write('%.3f' % confusion_matrix[i][i]);
            f.write('</td>\n');

            # Collect num_samples random paths to images of each type.
            # Training examples.
            train_examples = np.take(train_image_paths, np.argwhere(train_labels == cat))

            # True positives. There might not be enough of these if the classifier
            # is bad
            true_positives = np.take(test_image_paths,
                                     np.argwhere(np.logical_and(test_labels == cat, predicted_categories == cat)))

            # False positives.
            false_positive_inds = np.argwhere(
                np.logical_and(np.invert(cat == test_labels), cat == predicted_categories))
            false_positives = np.take(test_image_paths, false_positive_inds)
            false_positive_labels = np.take(test_labels, false_positive_inds)

            # False negatives.
            false_negative_inds = np.argwhere(
                np.logical_and(cat == test_labels, np.invert(cat == predicted_categories)))
            false_negatives = np.take(test_image_paths, false_negative_inds)
            false_negative_labels = np.take(predicted_categories, false_negative_inds)

            # Randomize each list of files
            np.random.shuffle(train_examples)
            np.random.shuffle(true_positives)
            # 保持误报图片和标签的对应关系
            rng_state = np.random.get_state()
            np.random.shuffle(false_positives)
            np.random.set_state(rng_state)
            np.random.shuffle(false_positive_labels)

            rng_state = np.random.get_state()
            np.random.shuffle(false_negatives)
            np.random.set_state(rng_state)
            np.random.shuffle(false_negative_labels)

            # Truncate each list to be at most num_samples long
            train_examples = train_examples[0:min(len(train_examples), num_samples)]
            true_positives = true_positives[0:min(len(true_positives), num_samples)]
            false_positives = false_positives[0:min(len(false_positives), num_samples)]
            false_positive_labels = false_positive_labels[0:min(len(false_positive_labels), num_samples)]
            false_negatives = false_negatives[0:min(len(false_negatives), num_samples)]
            false_negative_labels = false_negative_labels[0:min(len(false_negative_labels), num_samples)]

            # Sample training images
            # Create and save all of the thumbnails
            for j in range(num_samples):
                if j + 1 <= len(train_examples):
                    thisExample = train_examples[j][0]
                    tmp = skimage.io.imread(thisExample)

                    # 处理图像维度问题
                    tmp = process_image_dimensions(tmp)

                    # 获取调整后的尺寸
                    height, width = rescale(tmp.shape, thumbnail_height)

                    # 调整图像尺寸
                    tmp = resize_image_for_display(tmp, (height, width))

                    name = os.path.basename(thisExample)
                    tmp_uint8 = (tmp * 255).astype(np.uint8)
                    skimage.io.imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8)
                    f.write('<td bgcolor=LightBlue>')
                    f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height))
                    f.write('</td>\n')
                else:
                    f.write('<td bgcolor=LightBlue>')
                    f.write('</td>\n')

            for j in range(num_samples):
                if j + 1 <= len(true_positives):
                    thisExample = true_positives[j][0]
                    tmp = skimage.io.imread(thisExample)

                    # 处理图像维度问题
                    tmp = process_image_dimensions(tmp)

                    # 获取调整后的尺寸
                    height, width = rescale(tmp.shape, thumbnail_height)

                    # 调整图像尺寸
                    tmp = resize_image_for_display(tmp, (height, width))

                    name = os.path.basename(thisExample)
                    tmp_uint8 = (tmp * 255).astype(np.uint8)
                    skimage.io.imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
                    f.write('<td bgcolor=LightGreen>');
                    f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height))
                    f.write('</td>\n');
                else:
                    f.write('<td bgcolor=LightGreen>');
                    f.write('</td>\n');

            for j in range(num_samples):
                if j + 1 <= len(false_positives):
                    thisExample = false_positives[j][0]
                    tmp = skimage.io.imread(thisExample)

                    # 处理图像维度问题
                    tmp = process_image_dimensions(tmp)

                    # 获取调整后的尺寸
                    height, width = rescale(tmp.shape, thumbnail_height)

                    # 调整图像尺寸
                    tmp = resize_image_for_display(tmp, (height, width))

                    name = os.path.basename(thisExample)
                    tmp_uint8 = (tmp * 255).astype(np.uint8)
                    skimage.io.imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
                    f.write('<td bgcolor=LightCoral>');
                    f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height))
                    f.write('<br><small>%s</small>' % false_positive_labels[j][0]);
                    f.write('</td>\n');
                else:
                    f.write('<td bgcolor=LightCoral>');
                    f.write('</td>\n');

            for j in range(num_samples):
                if j + 1 <= len(false_negatives):
                    thisExample = false_negatives[j][0]
                    tmp = skimage.io.imread(thisExample)

                    # 处理图像维度问题
                    tmp = process_image_dimensions(tmp)

                    # 获取调整后的尺寸
                    height, width = rescale(tmp.shape, thumbnail_height)

                    # 调整图像尺寸
                    tmp = resize_image_for_display(tmp, (height, width))

                    name = os.path.basename(thisExample)
                    tmp_uint8 = (tmp * 255).astype(np.uint8)
                    skimage.io.imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
                    f.write('<td bgcolor=#FFBB55>');
                    f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height));
                    f.write('<br><small>%s</small>' % false_negative_labels[j][0]);
                    f.write('</td>\n');
                else:
                    f.write('<td bgcolor=#FFBB55>');
                    f.write('</td>\n');

            f.write('</tr>\n');

        f.write('<tr>\n');
        f.write('<th>Category name</th>\n');
        f.write('<th>Accuracy</th>\n');
        f.write('<th colspan=%d>Sample training images</th>\n' % num_samples);
        f.write('<th colspan=%d>Sample true positives</th>\n' % num_samples);
        f.write('<th colspan=%d>False positives with true label</th>\n' % num_samples);
        f.write('<th colspan=%d>False negatives with wrong predicted label</th>\n' % num_samples);
        f.write('</tr>\n');

        f.write('</table>\n');
        f.write('</center>\n\n\n');
        f.write('</div>\n');

        ## Create end of web page
        f.write('</body>\n');
        f.write('</html>\n');

    print('Wrote results page to results_webpage/index.html.')


def process_image_dimensions(image):
    """
    处理图像维度问题，确保图像是2D或3D

    参数:
        image: 输入的图像数组

    返回:
        processed_image: 处理后的2D或3D图像数组
    """
    # 处理4D图像
    if len(image.shape) == 4:
        if image.shape[0] == 1:
            image = image[0]  # 从 (1, h, w, c) 变为 (h, w, c)
        else:
            image = image[0]  # 取第一个

    # 处理RGBA图像（4通道）
    if len(image.shape) == 3 and image.shape[2] == 4:
        # 转换为RGB（忽略alpha通道）
        image = image[:, :, :3]

    # 处理单通道彩色图像（3D但第三维为1）
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    return image


def resize_image_for_display(image, output_shape):
    """
    安全地调整图像尺寸用于显示

    参数:
        image: 输入的图像数组
        output_shape: 输出形状 (height, width)

    返回:
        resized_image: 调整尺寸后的图像
    """
    height, width = output_shape

    if len(image.shape) == 2:
        # 灰度图像
        return resize(image, (height, width), anti_aliasing=True)
    elif len(image.shape) == 3:
        # 彩色图像
        return resize(image, (height, width), anti_aliasing=True, preserve_range=True)
    else:
        # 创建默认图像
        return np.zeros((height, width, 3), dtype=np.float32)


def rescale(dims, thumbnail_height):
    """
    计算缩略图尺寸，保持宽高比

    参数:
        dims: 图像形状元组
        thumbnail_height: 目标高度

    返回:
        (new_height, new_width): 新的高度和宽度
    """
    # 确保dims是有效的形状
    if len(dims) < 2:
        return (thumbnail_height, thumbnail_height)

    # 获取原始高度和宽度
    if len(dims) == 2:
        # 灰度图像: (height, width)
        original_height, original_width = dims[0], dims[1]
    elif len(dims) == 3:
        # 彩色图像: (height, width, channels)
        original_height, original_width = dims[0], dims[1]
    elif len(dims) == 4:
        # 4D图像: (batch, height, width, channels) 或类似
        original_height, original_width = dims[1], dims[2]
    else:
        return (thumbnail_height, thumbnail_height)

    # 计算缩放比例
    scale_factor = thumbnail_height / original_height

    # 计算新的宽度
    new_height = thumbnail_height
    new_width = int(round(original_width * scale_factor))

    return (new_height, new_width)


def create_results_webpage_simple(df, train_image_paths, train_labels,
                                  categories, abbr_categories, output_dir='results_webpage'):
    """
    简化的结果网页创建函数，只需要DataFrame和训练数据

    参数:
        df: 包含预测结果的DataFrame
        train_image_paths: 训练图像路径列表
        train_labels: 训练标签列表
        categories: 类别名称列表
        abbr_categories: 类别缩写列表
        output_dir: 输出目录
    """

    # 提取测试数据
    test_image_paths = df['图像路径'].values
    test_labels = df['标注类别名称'].values
    predicted_categories = df['top-1-预测名称'].values

    # 创建输出目录
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(os.path.join(output_dir, 'thumbnails')):
        os.makedirs(os.path.join(output_dir, 'thumbnails'))

    # 调用主函数
    create_results_webpage(train_image_paths, test_image_paths,
                           train_labels, test_labels,
                           categories, abbr_categories, predicted_categories)


