import cv2
import csv
import pyttsx3

# initialize video capture
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# create CSV file and write header
with open('eye_data.csv', mode='w') as csv_file:
    fieldnames = ['x', 'y', 'w', 'h', 'eye_x', 'eye_y']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# initialize text-to-speech engine
engine = pyttsx3.init()

# initialize variables to keep track of eye movement
last_x = None
last_y = None
last_direction = None

# Loop through the frames
while True:
    # Read the frame from the video capture object
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

        # loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            # calculate the position of the eye relative to the face
            eye_x = x + ex + ew // 2
            eye_y = y + ey + eh // 2

            # write x, y coordinates to CSV file
            with open('eye_data.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'x': x, 'y': y, 'w':w, 'h':h, 'eye_x': eye_x, 'eye_y': eye_y})

            # determine the direction of the eye movement
            direction = None
            if last_x is not None and last_y is not None:
                if eye_x > last_x:
                    direction = "right"
                elif eye_x < last_x:
                    direction = "left"
                else:
                    direction = "center"

            # update the last_x and last_y variables
            last_x = eye_x
            last_y = eye_y

            # display the direction of the eye movement
            if direction is not None and direction != last_direction:
                cv2.putText(frame, f"Looking {direction.capitalize()}", (eye_x, eye_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                last_direction = direction

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait


    # Wait for a key press to exit
    if cv2.waitKey(1)  == ord('q'):
        break

# Stop the engine and release video capture and close window
engine.stop()
cap.release()
cv2.destroyAllWindows()