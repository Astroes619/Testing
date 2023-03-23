import cv2
import csv

# initialize video capture
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# create CSV file and write header
with open('eye_tracking_data.csv', mode='w') as csv_file:
    fieldnames = ['x', 'y', 'w', 'h', 'direction']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# define gaze direction thresholds as constants
GAZE_LEFT_THRESHOLD = 0.4
GAZE_RIGHT_THRESHOLD = 0.6

# define function to get direction of gaze based on eye position
def get_gaze_direction(eye_x, eye_y, eye_w, eye_h, face_w):
    eye_center_x = eye_x + eye_w/2
    eye_center_y = eye_y + eye_h/2
    
    if eye_center_x < face_w * GAZE_LEFT_THRESHOLD:
        return 'left'
    elif eye_center_x > face_w * GAZE_RIGHT_THRESHOLD:
        return 'right'
    else:
        return 'center'

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

        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            # Get the direction of gaze
            gaze_direction = get_gaze_direction(ex, ey, ew, eh, w)

            # write data to CSV file
            with open('eye_tracking_data.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'x': x, 'y': y, 'w': w, 'h': h, 'direction': gaze_direction})

                # print data to terminal
                print(f"x: {x}, y: {y}, w: {w}, h: {h}, direction: {gaze_direction}")

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# release video capture and close window
cap.release()
cv2.destroyAllWindows()
