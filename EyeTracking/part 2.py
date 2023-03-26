import cv2
import csv
from collections import OrderedDict
import numpy as np

# initialize video capture
cap = cv2.VideoCapture(0)

# load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# create CSV file and write header
with open('eye_tracking_data.csv', mode='w') as csv_file:
    fieldnames = ['x', 'y', 'w', 'h', 'aspect_ratio', 'direction']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# initialize eye tracking dictionary
eye_trackers = OrderedDict()

# define function to get direction of gaze based on eye position
def get_gaze_direction(eye_x, eye_y, eye_w, eye_h, face_w):
    eye_center_x = eye_x + eye_w/2
    eye_center_y = eye_y + eye_h/2
    
    if eye_center_x < face_w*0.4:
        return 'left'
    elif eye_center_x > face_w*0.6:
        return 'right'
    else:
        return 'center'

# define variable to store previous eye coordinates
prev_eye_coords = None

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

        # write x, y coordinates to CSV file
        with open('eye_tracking_data.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'x': x, 'y': y, 'w':w, 'h':h})

        # Loop through each detected eye
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            # Check if this is the first iteration
            if prev_eye_coords is None:
                # Save the current eye coordinates as the previous coordinates
                prev_eye_coords = (eye_x, eye_y, eye_w, eye_h)
            else:
                # Calculate the difference between the current and previous eye coordinates
                diff = abs(np.array(prev_eye_coords) - np.array((eye_x, eye_y, eye_w, eye_h)))

                # Check if the difference is above a certain threshold
                if np.sum(diff) > 10:
                    # If the difference is above the threshold, assume that this is a false positive detection
                    # and continue to the next detected eye
                    continue

                # If the difference is below the threshold, update the previous eye coordinates
                prev_eye_coords = (eye_x, eye_y, eye_w, eye_h)

            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), (0, 0, 255), 2)

            # Calculate aspect ratio of eye region
            aspect_ratio = eye_w/eye_h

            # Get the direction of gaze
            gaze_direction = get_gaze_direction(eye_x, eye_y, eye_w, eye_h, w)

            # Write data to CSV file
            with open('eye_tracking_data.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'x': x, 'y': y, 'w': w, 'h': h, 'aspect_ratio': aspect_ratio, 'direction': gaze_direction})

                # Print data to terminal
                print(f"x: {x}, y: {y}, w: {w}, h: {h}, aspect_ratio: {aspect_ratio}, direction: {gaze_direction}")

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
