# Author: Vedant Singhania
# Machine Learning | Spring 2020
# University of Colorado Denver
# Final Project

import h5py
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import joblib
import matplotlib.pyplot as plt
from feature_extraction import HSV_histogram
from imblearn.over_sampling import SMOTE



fixed_size = tuple((500, 500))
warnings.filterwarnings('ignore')
smote = SMOTE('minority')

seed = 9
train_path = "dataset/train"
test_path = "dataset/images/test"
scoring = "accuracy"

# get the training labels
train_labels = ['Green', 'Midripe', 'Overripe', 'Yellowish_Green']

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', svm.SVC(random_state=seed, kernel='linear', C=10, probability=True)))
#models.append(('SVM', svm.SVC(random_state=seed, kernel='sigmoid')))
#models.append(('SVM', svm.SVC(random_state=seed, kernel='rbf')))
#models.append(('SVM', svm.SVC(random_state=seed, kernel='poly', degree=8)))
#models.append(('SVM', svm.SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    # decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    # max_iter=-1, probability=False, random_state=None, shrinking=True,
    # tol=0.001, verbose=False)))
# models.append(('SVM', svm.SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)))

# variables to hold the results and names
results = []
names = []

# import the feature vector and trained labels
hsv_features_h5  = h5py.File('output/data.h5', 'r')
hsv_labels_h5 = h5py.File('output/labels.h5', 'r')

global_features_string = hsv_features_h5['dataset_1']
global_labels_string = hsv_labels_h5['dataset_1']

hsv_features = np.array(global_features_string)
hsv_labels = np.array(global_labels_string)

hsv_features_h5.close()
hsv_labels_h5.close()

print("****Starting Training...****")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(hsv_features),
                                                                                          np.array(hsv_labels),
                                                                                          test_size=0.1,
                                                                                          random_state=seed)

print("Split training and testing data...")
print("Training data: ".format(trainDataGlobal.shape))
print("Testing data: ".format(testDataGlobal.shape))
print("Training labels: ".format(trainLabelsGlobal.shape))
print("Testing labels : ".format(testLabelsGlobal.shape))

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

print("****Starting testing....****")
######################################################
# Testing Section
######################################################

clf = svm.SVC(random_state=seed, kernel='linear', C=10, probability=True)
# clf = svm.SVC(random_state=seed, kernel='poly', degree=8)
#clf = svm.SVC(random_state=seed, kernel='sigmoid')
#clf = svm.SVC(random_state=seed, kernel='rbf')
# clf = svm.SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# clf = svm.SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

# SMOTE dataset
X_sm, y_sm = smote.fit_sample(trainDataGlobal, trainLabelsGlobal)
print(X_sm.shape, y_sm.shape)

# model fitting
#clf.fit(trainDataGlobal, trainLabelsGlobal)
clf.fit(X_sm, y_sm)

disp = metrics.plot_confusion_matrix(clf, testDataGlobal, testLabelsGlobal, display_labels=train_labels, cmap=plt.cm.Blues, normalize='true')
print(disp.confusion_matrix)
plt.show()

y_preds = clf.predict(testDataGlobal)
print('\n')
print(metrics.classification_report(testLabelsGlobal, y_preds, target_names=train_labels, output_dict=False))
print('\n')
print(metrics.classification_report(testLabelsGlobal, y_preds, target_names=train_labels, output_dict=True))

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)
    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Feature extraction
    ####################################
    fv_histogram = HSV_histogram(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = fv_histogram

    # scale the extracted HSV features
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_feature.reshape(-1, 1))
    # make predictions
    prediction = clf.predict(rescaled_feature.reshape(1, -1))[0]
    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = cv2.bilateralFilter(image, 7, sigmaSpace=75, sigmaColor=75)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    plt.show()
