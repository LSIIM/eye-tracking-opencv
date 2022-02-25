
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import pandas as pd

from face_adjustments_module import FaceAdjuster
from face_mesh_module import FaceMeshDetector
from eye_feature_detector_module import EyeModule
from definitions import *

def analyseFace(image, extractor):
    row, col = image.shape[:2]
    img_new_width = initial_image_width
    rt = img_new_width/col
    image = cv2.resize(image, (img_new_width, int(row*rt)))
    lms = extractor.findFaceMesh(image.copy())
    
    if(lms is None):
        print("/nNo faces")
        return [], "No Faces detected"
    
    # --------------------------------------------------------------------------
    adjuster = FaceAdjuster(image.copy(), lms)
    _, succeed = adjuster.alignEyes()
    if not succeed:
        return [], adjuster.error
    _, succeed = adjuster.alignFace()
    if not succeed:
        return [], adjuster.error
    _, succeed = adjuster.faceCrop()
    if not succeed:
        return [], adjuster.error
    _, succeed = adjuster.alignFace()
    if not succeed:
        return [], adjuster.error
    finalImage, succeed = adjuster.fixImageSizeWitBorders()
    if not succeed:
        return [], adjuster.error

    nlms = adjuster.getLms()

    return nlms, finalImage, None

def process_video(path = ""):
    camera = cv2.VideoCapture( str(path))
    vfps = (camera.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    name = path.split('/')
    name = name[len(name)-1].split('.')
    name = name[0]
    try:
        os.mkdir(prc_path+name )
    except:
        print("Diretorio ja existe")
    path = prc_path+ '/' + name + "/"
    name =  prc_path+"/"+name + '/video.avi'
    vLength = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    #file = open(path+"data.pickle", 'rb')
    (h,w) = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #out = cv2.VideoWriter(name, fourcc,int(vfps),(h,w))
    
    past_positions = {
        "left": [],
        "right": []
    }
    landmarks_extractor = FaceMeshDetector()
    for i in tqdm(range (0,vLength+1)):
        data = {
            "left_eye": {
                "iris":{
                    "x": None,
                    "y": None,
                    "r": None
                },
                "pupil":{
                    "x": None,
                    "y": None,
                    "r": None
                }
            },
            "right_eye": {
                "iris":{
                    "x": None,
                    "y": None,
                    "r": None
                },
                "pupil":{
                    "x": None,
                    "y": None,
                    "r": None
                }
            }
        }
        ret, frame = camera.read()
        if ret: 
            clear_image = frame.copy()
            try:
                lms, fimage, err = analyseFace(
                    clear_image, landmarks_extractor)
            except Exception as exception:
                err = type(exception).__name__
                print()
                print(err)
                continue


            # Identifies where the iris is and get its radius
            eye_module = EyeModule(image=fimage,data_save=data,lms=lms)
            left_iris, right_iris = eye_module.detect_iris()
            

            eye_module.detect_pupil()


            cv2.circle(fimage, left_iris[0],left_iris[1], (255,0,255), 1, cv2.LINE_AA)
            cv2.circle(fimage, right_iris[0],right_iris[1], (255,0,255), 1, cv2.LINE_AA)
            cv2.imshow("adjusted",fimage)
            cv2.waitKey(1)
            
if __name__=="__main__":
    process_video(path = raw_path+"/bebe.mp4")
    