import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    # img = detector.findHands(img, draw=False)
    # if i want to remove the tracking as well, eveything can be tweaked but it wont affect the orginal module
    lmList = detector.findPosition(img)
    # lmList = detector.findPosition(img, draw=False) # if i dont want the color from the
    # module i just have to make the draw false
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)