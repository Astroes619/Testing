import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from flask_socketio import emit

@socketio.on('frame')
def handle_frame(frame):
    # Decode the base64-encoded frame
    img_data = base64.b64decode(frame.split(',')[1])
    img = Image.open(BytesIO(img_data))
    img = np.array(img)

    # Convert the frame to grayscale and process it with OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through each detected face
    for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Detect the eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Loop through each detected eye
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around the eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

                # Your additional processing logic here

    # Encode the processed frame and emit it back to the client
    _, buffer = cv2.imencode('.jpg', img)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    emit('frame', {'frame': encoded_frame})
