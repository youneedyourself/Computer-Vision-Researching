import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess the images
def preprocess_images(images):
    processed_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        processed_images.append(descriptors)
    return np.vstack(processed_images)

# Preprocess the training and test images
train_descriptors = preprocess_images(train_images)
test_descriptors = preprocess_images(test_images)

# Use KMeans for clustering
kmeans = KMeans(n_clusters=100)
kmeans.fit(train_descriptors)

# Assign each descriptor to the nearest cluster center
train_features = kmeans.predict(train_descriptors)
test_features = kmeans.predict(test_descriptors)

# Use Bag of Visual Words (BoVW) representation
def bag_of_visual_words(features, n_clusters):
    hist = np.zeros(n_clusters)
    for item in features:
        hist[item] += 1
    return hist

train_hist = [bag_of_visual_words(features, 100) for features in train_features]
test_hist = [bag_of_visual_words(features, 100) for features in test_features]

# Train a classifier (SVM in this case)
classifier = make_pipeline(StandardScaler(), SVC())
classifier.fit(train_hist, train_labels)

# Make predictions on the test set
predictions = classifier.predict(test_hist)

# Evaluate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
