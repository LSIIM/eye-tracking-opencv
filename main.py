#-*- coding:utf-8 -*-
from FaceModule import FaceDetector
from EyeModule import iris_detection

import cv2 as cv
import numpy as np
from tqdm import tqdm
import os

def process_video(path,):
    '''cameraID = 0

    # creates a camera obj - WEBCAM
    camera = cv.VideoCapture(cameraID)
'''

    camera = cv.VideoCapture( str(path))
    vfps = (camera.get(cv.CAP_PROP_FPS))

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    face_detector = FaceDetector(maxNumFaces=1,minDetectionConfidence=0.9)

    name = path.split('/')
    name = name[len(name)-1].split('.')
    name = name[0]
    name =  './vds/prc/'+name + '.avi'
    (h,w) = int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(name, fourcc,int(vfps),(h,w))

    while True:
        ret, frame = camera.read()
        if ret: 
            clear_image = frame.copy()
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
                        #cv.imshow("L eye",np.array(eye_image))
                except:
                    print("No left eye on image")
                
                if eye_image != []:
                    id = iris_detection(np.array(eye_image))
                    x,y,r = id.start_detection()
                    if (x,y,r) != (0,0,0):
                        #print(x,y,r)
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
                        #cv.imshow("R eye",np.array(eye_image))
                except:
                    print("No right eye on image")
                if eye_image != []:
                    id = iris_detection(np.array(eye_image))
                    x,y,r = id.start_detection()
                    if (x,y,r) != (0,0,0):
                        #print(x,y,r)
                        cv.circle(frame, (x+left, y+top), r, (0, 255, 0), 2)


                cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
            #cv.imshow('Frame',frame)
            out.write(frame)
            '''key = cv.waitKey(1)

            if key == ord('q'):
                break'''
        else:
            break
    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    for video in tqdm(os.listdir('./vds/raw')):
        process_video("./vds/raw/"+video)