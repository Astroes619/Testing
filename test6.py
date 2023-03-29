import base64
from django.http import HttpResponse
from django.shortcuts import render
import cv2

def webcam(request):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert the frame to base64-encoded string
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode()

        # Pass the encoded image string to the HTML template
        context = {'image': img_str}
        return render(request, 'webcam.html', context)

    cap.release()
    cv2.destroyAllWindows()
