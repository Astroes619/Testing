import os
import csv
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError
import joblib

def extract_hog_features(image):
    image = cv2.resize(image, (100, 50))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False)
    return features

clf = joblib.load('trained_model.joblib')

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

def write_eye_tracking_data(data):
    with open('eye_tracking_data.csv', mode='a') as csv_file:
        fieldnames = ['x', 'y', 'w', 'h', 'aspect_ratio', 'ground_truth', 'predicted']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(data)


# Initialize cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the video frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the video frame

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                raise HttpResponseServerError("Error: Unable to read video frame.")

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

                    # Comment out the text display if not needed
                    # cv2.putText(frame, f"Ground Truth: {ground_truth}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    # cv2.putText(frame, f"Predicted: {predicted}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    eye_tracking_data.append({
                        'x': ex, 'y': ey, 'w': ew, 'h': eh, 'aspect_ratio': aspect_ratio,
                        'ground_truth': ground_truth, 'predicted': predicted
                    })

                    write_eye_tracking_data(eye_tracking_data[-1])  # Write the latest data point to the CSV file

                    print(f"x: {ex}, y: {ey}, w: {ew}, h: {eh}, aspect_ratio: {aspect_ratio}, ground_truth: {ground_truth}, predicted: {predicted}")

            ret, jpeg = cv2.imencode('.jpg', frame)

            if not ret:
                raise HttpResponseServerError("Error: Unable to encode video frame.")

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    finally:
        correct_predictions = 0
        total_predictions = len(eye_tracking_data)

        for data_point in eye_tracking_data:
            if data_point['ground_truth'] == data_point['predicted']:
                correct_predictions += 1

        final_accuracy = (correct_predictions / total_predictions) * 100
        print(f"Final Accuracy: {final_accuracy:.2f}%")

        cap.release()

def live_feed(request):
    try:
        frame_gen = gen_frames()
        for frame in frame_gen:
            if frame is None:
                raise ValueError("Error: Unable to read or encode video frame.")
            yield frame
    except ValueError as e:
        return HttpResponseServerError(str(e))


def dynamic_stream(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )

def index(request):
    return render(request, 'eyetracking_app/index.html')
