#-*- coding:utf-8 -*-
from itertools import count

from numpy.core.fromnumeric import shape
from FaceModule import FaceDetector
from EyeModule import iris_detection

import cv2 as cv
import numpy as np
from tqdm import tqdm
import os
from scipy import ndimage
import math
def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# faz os traços depois do crop
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
    vLength = int(camera.get(cv.CAP_PROP_FRAME_COUNT))
    #file = open(path+"data.pickle", 'rb')
    (w,h) = int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(name, fourcc,int(vfps),(h,w))
    
    face_detector = FaceDetector(maxNumFaces=1,minDetectionConfidence=0.9)
    past_positions = {
        "left": [],
        "right": []
    }
    for i in tqdm(range (0,vLength+1)):
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
            #print("Ori: ",clear_image.shape)
            #frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            frame,face1 = face_detector.findFaceMesh(frame)
            rotated = frame.copy()
            resized = frame.copy()
            top,left,bottom,right = 0,0,0,0
            if(len(face1)>0):
                face1 = face1[0]
                top,left,bottom,right = face_detector.find_face_border(face1)
                #cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                face_centralized_image = []
                if top<0:
                    top = 0
                if left<0:
                    left = 0
                
                for j in range(top,bottom):
                    row = []
                    if(j>=h):
                        continue
                    for i in range(left,right):
                        if(i>=w):
                            continue
                        row.append(clear_image[j][i])
                    face_centralized_image.append(row)
                face_centralized_image = np.array(face_centralized_image)

                topL,_,_,rightL = face_detector.find_l_eye_border(face1)
                topR,_,_,rightR = face_detector.find_r_eye_border(face1)
                
                rotated = ndimage.rotate(clear_image, (math.atan((topR-topL)/(rightR-rightL))*180/3.14))

                
                

                frame,face2 = face_detector.findFaceMesh(rotated)
                top,left,bottom,right = 0,0,0,0
                if(len(face2)>0):
                    clear_image = rotated
                    face2 = face2[0]
                    top,left,bottom,right = face_detector.find_face_border(face2)
                    #cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                    face_centralized_image = []
                    if top<0:
                        top = 0
                    if left<0:
                        left = 0
                    if bottom>clear_image.shape[0]:
                        bottom = clear_image.shape[0]
                    if right>clear_image.shape[1]:
                        right = clear_image.shape[1]
                    
                    for j in range(top,bottom):
                        row = []
                        for i in range(left,right):
                            row.append(clear_image[j][i])
                        face_centralized_image.append(row)
                    face_centralized_image = np.array(face_centralized_image)

                    h_dif = (h/face_centralized_image.shape[0])
                    w_dif = (w/face_centralized_image.shape[1])
                    print("\nDIFS: ",h_dif,w_dif)
                    if(h_dif>w_dif):
                        resized =cv.resize(face_centralized_image,(int(face_centralized_image.shape[0]*w_dif),int(face_centralized_image.shape[1]*w_dif)))
                        
                    else:
                        resized =cv.resize(face_centralized_image,(int(face_centralized_image.shape[0]*h_dif),int(face_centralized_image.shape[1]*h_dif)))
                    print("Shape: ",resized.shape)
                    print(h,w)
                    
                    print((0,h-resized.shape[0]),(0,w-resized.shape[1]))
                    resized = np.pad(resized, [(0,h-resized.shape[0]),(0,w-resized.shape[1]),(0,0)],mode="constant")
                
            
                    frame,face3 = face_detector.findFaceMesh(resized,True)
                    top,left,bottom,right = 0,0,0,0
                    if(len(face3)>0):
                        clear_image = resized
                        face3 = face3[0]
                        '''----------------------------------------------------------
                                            LEFT EYE PROCESSING
                        ----------------------------------------------------------'''
                        top,left,bottom,right = face_detector.find_l_eye_border(face3)
                        cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
                        cv.imshow("Resized",frame)
                        eye_image = []
                        try:
                            if top>=0 and left>=0 and bottom>=0 and right>=0:
                                for j in range(top,bottom):
                                    row = []
                                    for i in range(left,right):
                                        row.append(clear_image[j][i])
                                    eye_image.append(row)
                                #print("here")
                                data['left_eye']['raw'] = np.array(eye_image)        
                                cv.imshow("EYE",np.array(eye_image)    )                
                        except:
                            print("No left eye on image")
                            past_positions['left'] = []
                        
                        if eye_image != []:
                            id = iris_detection(image = np.array(eye_image),data_save=data)
                            x,y,r,data = id.start_detection()
                            if (x,y,r) != ([0],[0],[0]):
                                #print(x,y,r)
                                
                                past_positions["left"].append([x[0],y[0]])
                                data['left_eye']['x'] = x[0]
                                data['left_eye']['y'] = y[0]
                                #for i in range(len(x)):
                                    #cv.ellipse(frame, (x[i]+left, y[i]+top), r[i],0,0,360, (0, 255, 0), 2)
                            else:
                                past_positions['left'] = []
                        else:
                            past_positions['left'] = []

                        '''----------------------------------------------------------
                                            Right EYE PROCESSING
                        ----------------------------------------------------------'''
                        top,left,bottom,right = face_detector.find_r_eye_border(face3)
                        cv.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
                        
                        eye_image = []
                        try:
                            if top>=0 and left>=0 and bottom>=0 and right>=0:
                                for j in range(top,bottom):
                                    row = []
                                    for i in range(left,right):
                                        row.append(clear_image[j][i])
                                    eye_image.append(row)
                                data['right_eye']['raw'] = np.array(eye_image)
                                
                        except:
                            print("No right eye on image")
                            past_positions['right'] = []
                        if eye_image != []:
                            id = iris_detection(image = np.array(eye_image),data_save=data)
                            x,y,r,data = id.start_detection()
                            if (x,y,r) != ([0],[0],[0]):
                                past_positions["right"].append([x[0],y[0]])
                                data['right_eye']['x'] = x[0]
                                data['right_eye']['y'] = y[0]
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
                else:
                    past_positions['left'] = []
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

            
            
            
            cv.imshow("out",frame)
            cv.waitKey(1)

            out.write(frame)

            
        else:
            break
    camera.release()
    cv.destroyAllWindows()
        

if __name__ == "__main__":
    for video in tqdm(os.listdir('./vds/raw')):
            process_video(path = "./vds/raw/"+video)
            os.system("cls")