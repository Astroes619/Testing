import cv2
import pandas as pd
import os

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')




# Create an empty DataFrame to store the eye positions
df = pd.DataFrame(columns=['Left Eye X', 'Left Eye Y', 'Right Eye X', 'Right Eye Y'])

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


        # Loop through each detected eye
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            # Get the position of the eye relative to the face
            eye_x = x + ex + ew/2
            eye_y = y + ey + eh/2

            # Store the eye position in the DataFrame
            if i == 0:
                df.loc[len(df)] = [eye_x, eye_y, None, None]
            else:
                df.loc[len(df)-1, 'Right Eye X'] = eye_x
                df.loc[len(df)-1, 'Right Eye Y'] = eye_y

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Save the DataFrame to an Excel file
df.to_excel('eye_tracking_data.xlsx', index=False)
print(pd.__file__)

print(os.getcwd())
