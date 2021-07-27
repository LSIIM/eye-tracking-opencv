#-*- coding:utf-8 -*-
from FaceModule import FaceDetector
from EyeModule import iris_detection

import cv2 as cv
import numpy as np
from tqdm import tqdm
import os
import pickle


def process_video(path = ""):
    
    camera = cv.VideoCapture( str(path))
    vfps = (camera.get(cv.CAP_PROP_FPS))

    fourcc = cv.VideoWriter_fourcc(*'XVID')

    name = path.split('/')
    name = name[len(name)-1].split('.')
    name = name[0]
    try:
        os.mkdir('./vds/nir/'+name )
    except:
        print("Diretorio ja existe")
    path = './vds/nir/'+name +"/"
    name =  './vds/nir/'+name + '/video.avi'
    
    #file = open(path+"data.pickle", 'rb')
    (h,w) = int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(name, fourcc,int(vfps),(h,w))
    
    face_detector = FaceDetector(maxNumFaces=1,minDetectionConfidence=0.9)
    past_positions = {
        "left": [],
        "right": []
    }
    while True:
        data = {
            "left_eye": {
                "raw": None,
                "blurred": None,
                "binary": None,
                "histograms":{
                    "rows": None,
                    "columns": None
                },
                "x": 0,
                "y":0
            },
            "right_eye": {
                "raw": None,
                "blurred": None,
                "binary": None,
                "histograms":{
                    "rows": None,
                    "columns": None
                },
                "x": 0,
                "y":0
            }
        }
        ret, frame = camera.read()
        if ret: 
            clear_image = frame.copy()
            #frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            frame,face = face_detector.findFaceMesh(frame)
            top,left,bottom,right = 0,0,0,0
            if(len(face)>0):
                face = face[0]
                top,left,bottom,right = face_detector.find_face_border(face)
                #cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                face_centralized_image = []
                if top<0:
                    top = 0
                if left<0:
                    left = 0
                
                for j in range(top,bottom):
                    row = []
                    if(j>=clear_image.shape[0]):
                        continue
                    for i in range(left,right):
                        if(i>=clear_image.shape[1]):
                            continue
                        row.append(clear_image[j][i])
                    face_centralized_image.append(row)
                face_centralized_image = np.array(face_centralized_image)
                frame = face_centralized_image
                
                off_top,off_left,off_bottom,off_right = top,left,bottom,right
                '''----------------------------------------------------------
                                    LEFT EYE PROCESSING
                ----------------------------------------------------------'''
                top,left,bottom,right = face_detector.find_l_eye_border(face)
                top,left,bottom,right = top - off_top,left - off_left,bottom - off_top,right - off_left
                cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
                eye_image = []
                try:
                    if top>=0 and left>=0 and bottom>=0 and right>=0:
                        for j in range(top,bottom):
                            row = []
                            for i in range(left,right):
                                row.append(face_centralized_image[j][i])
                            eye_image.append(row)
                        #print("here")
                        data['left_eye']['raw'] = np.array(eye_image)                        
                except:
                    print("No left eye on image")
                    past_positions['left'] = []
                
                if eye_image != []:
                    id = iris_detection(image = np.array(eye_image),data_save=data)
                    x,y,r,data = id.start_detection()
                    if (x,y,r) != ([0],[0],[0]):
                        #print(x,y,r)
                        
                        past_positions["left"].append([x[0]+left,y[0]+top])
                        data['left_eye']['x'] = x[0]+left
                        data['left_eye']['y'] = y[0]+top
                        #for i in range(len(x)):
                            #cv.ellipse(frame, (x[i]+left, y[i]+top), r[i],0,0,360, (0, 255, 0), 2)
                    else:
                        past_positions['left'] = []
                else:
                    past_positions['left'] = []

                '''----------------------------------------------------------
                                    Right EYE PROCESSING
                ----------------------------------------------------------'''
                top,left,bottom,right = face_detector.find_r_eye_border(face)
                top,left,bottom,right = top - off_top,left - off_left,bottom - off_top,right - off_left
                cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
                
                eye_image = []
                try:
                    if top>=0 and left>=0 and bottom>=0 and right>=0:
                        for j in range(top,bottom):
                            row = []
                            for i in range(left,right):
                                row.append(face_centralized_image[j][i])
                            eye_image.append(row)
                        data['right_eye']['raw'] = np.array(eye_image)
                        
                except:
                    print("No right eye on image")
                    past_positions['right'] = []
                if eye_image != []:
                    id = iris_detection(image = np.array(eye_image),data_save=data)
                    x,y,r,data = id.start_detection()
                    if (x,y,r) != ([0],[0],[0]):
                        past_positions["right"].append([x[0]+left,y[0]+top])
                        data['right_eye']['x'] = x[0]+left
                        data['right_eye']['y'] = y[0]+top
                        #print(x,y,r)
                        #for i in range(len(x)):
                            #cv.ellipse(frame, (x[i]+left, y[i]+top), r[i],0,0,360, (0, 255, 0), 2)
                    else:
                        past_positions['right'] = []
                else:
                    past_positions['right'] = []

                
            else:
                past_positions['left'] = []
                past_positions['right'] = []



            # Desenha as linhas do olho direito
            for i in range(1,len(past_positions["right"])):
                end_point=past_positions["right"][i]
                start_point=past_positions["right"][i-1]
                if i == len(past_positions["right"])-1:
                    color = (255,0,0)
                else:
                    color = (0,0,255)
                thickness= int(4*i/len(past_positions["right"]))+1
                #print(thickness)
                #cv.putText(frame,'.',(pos),cv.FONT_HERSHEY_PLAIN,thickness,color,1)
                frame = cv.line(frame, start_point, end_point, color, thickness)

            # Desenha as linhas do olho esquerdo
            for i in range(1,len(past_positions["left"])):
                end_point=past_positions["left"][i]
                start_point=past_positions["left"][i-1]
                if i == len(past_positions["left"])-1:
                    color = (255,0,0)
                else:
                    color = (0,0,255)
                thickness =int(4*i/len(past_positions["left"]))+1
                #cv.putText(frame,'.',(pos),cv.FONT_HERSHEY_PLAIN,thickness,color,1)
                frame = cv.line(frame, start_point, end_point, color, thickness)

            
            
            #print(frame[20][20][1])
            #print([(0,clear_image.shape[0]-frame.shape[0]),(0,clear_image.shape[1]-frame.shape[1]),(0,0)])
            frame = np.pad(frame, [(0,clear_image.shape[0]-frame.shape[0]),(0,clear_image.shape[1]-frame.shape[1]),(0,0)],mode="constant")

            out.write(frame)

            
        else:
            break
    camera.release()
    cv.destroyAllWindows()
        

if __name__ == "__main__":
    for video in tqdm(os.listdir('./vds/raw')):
            process_video(path = "./vds/raw/"+video)