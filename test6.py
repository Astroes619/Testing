import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render

# Define the video capture object
cap = cv2.VideoCapture(0)

# Define the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the view function
def webcam(request):
    # Define the generator function that will capture frames from the video stream
    def generate_frames():
        while True:
            # Capture a frame from the video stream
            ret, frame = cap.read()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Crop the face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detect the eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)

                # Loop through each detected eye
                for (ex, ey, ew, eh) in eyes:
                    # Draw a rectangle around the eye
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)

            # Yield the buffer as bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # Return a streaming HTTP response with the video frames
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

    cv2.imshow('frame', frame)

    if cv2.waitKey(1)  == ord('q'):
        break

