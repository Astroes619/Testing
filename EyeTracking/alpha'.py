
import cv2

import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap1 =  cv2.VideoCapture(0)

while True:
        success, vid = cap1.read()


        imgRGB = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)

        results = pose.process(imgRGB)

        if results.pose_landmarks:

            mpDraw.draw_landmarks(vid, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = vid.shape
                cx,cy = int(lm.x * w), int( lm.y * h)
                print(id,"\nX:", cx, " \nY:", cy)



        cv2.imshow("vid1",vid)

        cv2.waitKey(1)

