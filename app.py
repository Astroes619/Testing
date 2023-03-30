import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
from io import BytesIO
from PIL import Image

# Add these imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize an empty list to store data
eye_tracking_data = []

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

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)

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

                eye_tracking_data.append({'x': x, 'y': y, 'w': w, 'h': h, 'aspect_ratio': aspect_ratio, 'direction': gaze_direction})

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield frame

# Create a global variable to store the generator object
frame_generator = None

@socketio.on('connect', namespace='/video')
def connect():
    global frame_generator
    print('Client connected')
    frame_generator = gen_frames()
    emit('start', {'data': 'Connected'})

@socketio.on('frame', namespace='/video')
def frame():
    global frame_generator
    while True:
        frame = next(frame_generator)
        img_str = base64.b64encode(frame).decode('utf-8')
        emit('response', {'image': img_str})
        time.sleep(0.03)  # Adjust this value to control the frame rate



if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)  # Change the port number to an available one


