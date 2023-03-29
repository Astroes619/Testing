import cv2
import csv
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Add the extract_hog_features function and the training code from the previous response here

def extract_hog_features(image):
    image = cv2.resize(image, (100, 50))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False)
    return features

# Load the data from the CSV file
data = []

with open('eye_tracking_data.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        image_path = row[0]
        label = row[1]
        data.append([image_path, label])



# Load images and labels from the folders
eye_images = []
labels = []

for image_path, label in data:
    eye_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features = extract_hog_features(eye_image)
    eye_images.append(features)
    labels.append(label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(eye_images, labels, test_size=0.2)

# Create an SVM classifier and fit it to the training data
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Test the classifier on the testing data and print the accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Initialize video capture
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a CSV file to store the predicted gaze directions
with open('predicted_gaze_directions.csv', mode='w') as csv_file:
    fieldnames = ['gaze_direction']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# Loop through the frames
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            features = extract_hog_features(eye_img)

            # Predict gaze direction
            predicted_gaze_direction = clf.predict([features])[0]

            # Write the predicted gaze direction to the CSV file
            with open('predicted_gaze_directions.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'gaze_direction': predicted_gaze_direction})

            print(f"Predicted gaze direction: {predicted_gaze_direction}")

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
