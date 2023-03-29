import os
import cv2
import csv
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def extract_hog_features(image):
    image = cv2.resize(image, (100, 50))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False)
    return features

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

X_train, X_test, y_train, y_test = train_test_split(eye_images, labels, test_size=0.2)

clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_gaze_direction(eye_x, eye_y, eye_w, eye_h, face_w):
    eye_center_x = eye_x + eye_w/2
    eye_center_y = eye_y + eye_h/2
    
    left_range = face_w * 0.35
    right_range = face_w * 0.65
    center_range_left = face_w * 0.325
    center_range_right = face_w * 0.675

    if eye_center_x < left_range:
        return 'left'
    elif eye_center_x > right_range:
        return 'right'
    elif center_range_left < eye_center_x < center_range_right:
        return 'center'
    else:
        return 'unknown'

eye_tracking_data = []

with open('eye_tracking_data.csv', mode='a') as csv_file:
    fieldnames = ['x', 'y', 'w', 'h', 'aspect_ratio', 'ground_truth', 'predicted']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
   
     # Only write the header if the file is empty
    if csv_file.tell() == 0:
        writer.writeheader()
    
    writer.writerows(eye_tracking_data)

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
            eye_img_resized = cv2.resize(eye_img, (100, 50))

            aspect_ratio = ew/eh

            ground_truth = get_gaze_direction(ex, ey, ew, eh, w)

            features = extract_hog_features(eye_img_resized)
            predicted = clf.predict([features])[0]

            # cv2.putText(frame, f"Ground Truth: {ground_truth}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(frame, f"Predicted: {predicted}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            eye_tracking_data.append({
                'x': x, 'y': y, 'w': w, 'h': h, 'aspect_ratio': aspect_ratio,
                'ground_truth': ground_truth, 'predicted': predicted
            })

            print(f"x: {x}, y: {y}, w: {w}, h: {h}, aspect_ratio: {aspect_ratio}, ground_truth: {ground_truth}")

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

correct_predictions = 0
total_predictions = len(eye_tracking_data)

for data_point in eye_tracking_data:
    if data_point['ground_truth'] == data_point['predicted']:
        correct_predictions += 1

final_accuracy = (correct_predictions / total_predictions) * 100
print(f"Final Accuracy: {final_accuracy:.2f}%")

with open('eye_tracking_data.csv', mode='a') as csv_file:
    fieldnames = ['x', 'y', 'w', 'h', 'aspect_ratio', 'ground_truth', 'predicted']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writerows(eye_tracking_data)



cap.release()
cv2.destroyAllWindows()

