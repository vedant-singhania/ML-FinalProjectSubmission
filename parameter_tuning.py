# Author: Vedant Singhania
# Machine Learning | Spring 2020
# University of Colorado Denver
# Final Project

import h5py
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

fixed_size = tuple((500, 500))
warnings.filterwarnings('ignore')

num_trees = 100
test_size = 0.10
seed = 9
train_path = "dataset/train"
test_path = "dataset/images/test"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
scoring = "accuracy"

# get the training labels
train_labels = ['Green', 'Midripe', 'Overripe', 'Yellowish_Green']

# variables to hold the results and names
results = []
names = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()


# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10, 'scale']}
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, verbose=1)
# Train the classifier
clf_grid.fit(trainDataGlobal, trainLabelsGlobal)
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
