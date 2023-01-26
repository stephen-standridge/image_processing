import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# load the training set of images
train_images = []
train_labels = []
for i in range(1, 11):
    img = cv2.imread('train_image_' + str(i) + '.jpg')
    train_images.append(img)
    train_labels.append('image_' + str(i))

# extract features from the training set images
train_features = []
for img in train_images:
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    train_features.append(hist)

# create and train a k-nearest neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels)

# load the test image
test_img = cv2.imread('test_image.jpg')

# extract features from the test image
test_hist = cv2.calcHist([test_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
test_hist = cv2.normalize(test_hist, test_hist).flatten()

# predict the label of the test image
prediction = knn.predict([test_hist])

# check if the test image is unique compared to the training set
if prediction[0] == 'unknown':
    print("The test image is unique compared to the training set.")
else:
    print("The test image is not unique compared to the training set.")
