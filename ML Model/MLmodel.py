import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Function to extract HOG features from an image
def extract_hog_features(image):
    image = cv2.resize(image, (100, 50))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False)
    return features

def load_eye_images():
    eye_images = []
    labels = []

    directions = ['left', 'center', 'right']
    for direction in directions:
        folder_path = f'eye_images/{direction}'
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            features = extract_hog_features(img)
            eye_images.append(features)
            labels.append(direction)

    return eye_images, labels

def train_classifier():
    eye_images, labels = load_eye_images()

    np.random.seed(32)
    X_train, X_test, y_train, y_test = train_test_split(eye_images, labels, test_size=0.2)

    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return clf

classifier = train_classifier()
