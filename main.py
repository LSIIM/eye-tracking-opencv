#-*- coding:utf-8 -*-
from FaceModule import FaceDetector
import cv2 as cv
import numpy as np

cameraID = 0

# creates a camera obj
camera = cv.VideoCapture(cameraID)
face_detector = FaceDetector(maxNumFaces=1,minDetectionConfidence=0.9)
while True:
    ret, frame = camera.read()
    if ret:
        #frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img2,face = face_detector.findFaceMesh(frame)
        if(len(face)>=1):
            top,left,bottom,right = face_detector.find_face_border(face[0])
            cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        cv.imshow('Frame',frame)

        key = cv.waitKey(1)

        if key == ord('q'):
            breaK
camera.release()
cv.destroyAllWindows()