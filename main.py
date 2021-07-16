#-*- coding:utf-8 -*-
from FaceModule import FaceDetector
from EyeModule import iris_detection

import cv2 as cv
import numpy as np

cameraID = 0

# creates a camera obj
camera = cv.VideoCapture(cameraID)
face_detector = FaceDetector(maxNumFaces=1,minDetectionConfidence=0.9)
while True:
    ret, frame = camera.read()
    clear_image = frame.copy()
    if ret:
        #frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        frame,face = face_detector.findFaceMesh(frame)
        top,left,bottom,right = 0,0,0,0
        if(len(face)>0):
            face = face[0]
            top,left,bottom,right = face_detector.find_face_border(face)
            cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)


            '''----------------------------------------------------------
                                LEFT EYE PROCESSING
            ----------------------------------------------------------'''
            top,left,bottom,right = face_detector.find_l_eye_border(face)
            cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
            eye_image = []
            try:
                if top>=0 and left>=0 and bottom>=0 and right>=0:
                    for j in range(top,bottom):
                        row = []
                        for i in range(left,right):
                            row.append(clear_image[j][i])
                        eye_image.append(row)
                    cv.imshow("L eye",np.array(eye_image))
            except:
                print("No left eye on image")
            
            if eye_image != []:
                id = iris_detection(np.array(eye_image))
                x,y,r = id.start_detection()
                if (x,y,r) != (0,0,0):
                    print(x,y,r)
                    cv.circle(frame, (x+left, y+top), r, (0, 255, 0), 2)


            '''----------------------------------------------------------
                                Right EYE PROCESSING
            ----------------------------------------------------------'''
            top,left,bottom,right = face_detector.find_r_eye_border(face)
            eye_image = []
            try:
                if top>=0 and left>=0 and bottom>=0 and right>=0:
                    for j in range(top,bottom):
                        row = []
                        for i in range(left,right):
                            row.append(clear_image[j][i])
                        eye_image.append(row)
                    cv.imshow("R eye",np.array(eye_image))
            except:
                print("No right eye on image")
            if eye_image != []:
                id = iris_detection(np.array(eye_image))
                x,y,r = id.start_detection()
                if (x,y,r) != (0,0,0):
                    print(x,y,r)
                    cv.circle(frame, (x+left, y+top), r, (0, 255, 0), 2)


            cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
        cv.imshow('Frame',frame)

        key = cv.waitKey(1)

        if key == ord('q'):
            break
camera.release()
cv.destroyAllWindows()