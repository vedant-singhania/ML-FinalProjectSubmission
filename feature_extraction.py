# Author: Vedant Singhania
# Machine Learning | Spring 2020
# University of Colorado Denver
# Final Project

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import h5py

train_path = "dataset/images/set1_train"

# HSV Histogram
def HSV_histogram(image, mask=None):
    # smoothing image - very important
    image = cv2.bilateralFilter(image, 7, sigmaSpace=75, sigmaColor=75)
    # image converted from RGB to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image resized
    image = cv2.resize(image, tuple((500, 500)))
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()
    # histogram gets returned as feature vector

# empty lists to hold feature vectors and labels
global_features = []
labels = []
# Classes
train_labels = ['Green', 'Midripe', 'Overripe', 'Yellowish_Green']

# loop over images
images_in_class = 0
prefix = ''
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    if(training_name == "Green"):
        images_in_class = 100
        prefix = "g"
    if (training_name == "Midripe"):
        images_in_class = 84
        prefix = "m"
    if (training_name == "Overripe"):
        images_in_class = 29
        prefix = "v"
    if (training_name == "Yellowish_Green"):
        images_in_class = 44
        prefix = "y"
    # check images in each folder
    for x in range(1, images_in_class+1):
        file = dir + '/' + str(prefix) + str(x).zfill(3) + ".jpg"
        #print(file)
        #img = cv2.imread("dataset/images/set1_train/Green/g001.jpg")
        # Read in images
        img = cv2.imread(file)
        fv_histogram = HSV_histogram(img)

        global_feature = fv_histogram
        labels.append(current_label)
        global_features.append(global_feature)

    print("****Folder pre-processing complete: ".format(current_label))

print("**** Feature Extraction Complete.****")

# label class variables
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

# scale the extracted HSV features
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
#print(rescaled_features)

# hsv features and labels stored in h5 instead of csv files
hsv_features_h5 = h5py.File('output/data.h5', 'w')
hsv_features_h5.create_dataset('dataset_1', data=np.array(rescaled_features))
hsv_labels_h5 = h5py.File('output/labels.h5', 'w')
hsv_labels_h5.create_dataset('dataset_1', data=np.array(target))
hsv_features_h5.close()
hsv_labels_h5.close()

print("***Training Complete***")
