import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k = 1):
    '''

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

   对于每张测试图片，在训练集中找最相似的K张图片，看这些"邻居"大多数是什么类别，就预测为那个类别。k=1为最近邻算法
   1. 计算距离：与所有训练图片的特征距离
2. 找邻居：选出距离最近的K张训练图片
3. 投票决策：统计K个邻居的类别，最多的那个就是预测结果
    '''

    # 0) Gets the distance between each test image feature and each train image feature
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')
    # 1) Find the k closest features to each test image feature in euclidean space
    predictions = []

    for distance in distances:
        k_small_dis_labels = []
        sorted_dis_index = np.argsort(distance)

        # 2) Determine the labels of those k features
        for i in range(k):
            k_small_dis_labels.append(train_labels[sorted_dis_index[i]]) #获取k个最近邻居的标签
        # 3) Pick the most common label from the k
        most_common_label = Counter(k_small_dis_labels).most_common(1)[0][0]
        # 4) Store that label in a list
        predictions.append(most_common_label)

    return np.array(predictions)