import datetime
import logging
import os
import csv
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError, JsonResponse
import joblib
from firebase import bucket

from django.views.decorators.csrf import csrf_exempt
from firebase_admin import storage

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES['file']
        file_name = file.name

        # Upload the file to Firebase Storage
        blob = storage.bucket().blob(file_name)
        blob.upload_from_string(file.read())

        # Get the public URL of the uploaded file
        file_url = blob.public_url

        return JsonResponse({'file_url': file_url})
    else:
        return JsonResponse({'error': 'Invalid request method'})





# debugging ML 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('debug.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


logger.debug("Loading the trained model from: %s", "eyetracking_app/trained_model.joblib")

try:
    clf = joblib.load("eyetracking_app/trained_model.joblib")
    logger.debug("Trained model loaded successfully.")
except Exception as e:
    logger.error("Failed to load the trained model: %s", str(e))

logger.debug("Received frame for eye detection.")
logger.debug("Generating frames for video streaming.")
logger.debug("Serving dynamic video stream.")


recording = False
video_capture = cv2.VideoCapture(0)

def toggle_recording(request):
    global recording, video_capture

    if request.method == 'POST':
        recording = not recording

        if recording:
            # Start recording
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             # Generate a unique file name using a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f'output_{timestamp}.mp4'

            out = cv2.VideoWriter(file_name, fourcc, 20.0, (640, 480))

            while recording:
                ret, frame = video_capture.read()

                if not ret or frame is None:
                    continue

                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()
        else:
            # Stop recording
            video_capture.release()
            cv2.destroyAllWindows()

        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request'})


def extract_hog_features(image):
    image = cv2.resize(image, (100, 50))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False, channel_axis=-1)
    return features

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path for the trained model file in the same directory as the script
trained_model_path = os.path.join(current_script_dir, 'trained_model.joblib')

# Load the model
clf = joblib.load(trained_model_path)


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
