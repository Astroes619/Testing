import cv2
import os
import csv

# Initialize video capture
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a directory to save eye images
if not os.path.exists('eye_images'):
    os.makedirs('eye_images')

# Create directories for left, center, and right gazes
directions = ['left', 'center', 'right']
for direction in directions:
    if not os.path.exists(f'eye_images/{direction}'):
        os.makedirs(f'eye_images/{direction}')

# Create a CSV file to store the image paths and labels
with open('eye_tracking_data.csv', mode='a') as csv_file:
    fieldnames = ['image_path', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

img_count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for idx, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            # Save the preprocessed eye image
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (100, 50))

            cv2.imshow(f'eye_{idx}', eye_img)

            # Manually label the data as 'left', 'center', or 'right'
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break

            if key == ord('l'):  # Left
                label = 'left'
            elif key == ord('c'):  # Center
                label = 'center'
            elif key == ord('r'):  # Right
                label = 'right'
            else:
                continue

            img_path = f'eye_images/{label}/eye_{img_count}_idx_{idx}.png'
            cv2.imwrite(img_path, eye_img)

            # Write the image path and label to the CSV file
            with open('eye_tracking_data.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'image_path': img_path, 'label': label})

    cv2.imshow('frame', frame)

    if cv2.waitKey(100) == ord('q'):
        break

    img_count += 1

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
