import cv2
import csv
import numpy as np
from django.shortcuts import render

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


def webcam(request):
    cap = cv2.VideoCapture(0)
    eye_tracking_data = []

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
                eye_img = cv2.resize(eye_img, (100, 50))

                aspect_ratio = ew/eh

                gaze_direction = get_gaze_direction(ex, ey, ew, eh, w)

                # Append data to the list instead of writing to the CSV file
                eye_tracking_data.append({'x': x, 'y': y, 'w': w, 'h': h, 'aspect_ratio': aspect_ratio, 'direction': gaze_direction})

                print(f"x: {x}, y: {y}, w: {w}, h: {h}, aspect_ratio: {aspect_ratio}, direction: {gaze_direction}")

        cv2.imshow('frame', frame)

        if cv2.waitKey(1)  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Load the data from the list
    data = []
    labels = []

    for row in eye_tracking_data:
        data.append([float(row['x']), float(row['y']), float(row['w']), float(row['h']), float(row['aspect_ratio'])])
        labels.append(row['direction'])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Release video capture and close window
    cap.release()
    cv2.destroyAllWindows()
