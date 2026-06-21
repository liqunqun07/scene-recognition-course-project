from sklearn.svm import LinearSVC

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    l_svc = LinearSVC(random_state=0, tol=1e-5) #收敛容忍度（tolerance）

    # train LinearSVC model
    l_svc.fit(train_image_feats, train_labels) #训练
    # make prediction
    predictions = l_svc.predict(test_image_feats)#用来预测

    return predictions