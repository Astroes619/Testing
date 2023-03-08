import cv2
import csv

# initialize video capture
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# create CSV file and write header
with open('eye_tracking_data.csv', mode='w') as csv_file:
    fieldnames = ['x', 'y', 'w', 'h']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()


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

        print(x, y, w, h)

        # write x, y coordinates to CSV file
        with open('eye_tracking_data.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'x': x, 'y': y, 'w':w, 'h':h})


        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1)  == ord('q'):
        break


# release video capture and close window
cap.release()
cv2.destroyAllWindows()
