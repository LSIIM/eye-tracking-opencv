
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import pandas as pd


from positions_module import EyeDataModule, PositionsModule
from face_adjustments_module import FaceAdjuster
from face_mesh_module import FaceMeshDetector
from eye_feature_detector_module import EyeModule
from definitions import *
from drawing_utils import *

def adjustFace(image, extractor):
    row, col = image.shape[:2]
    img_new_width = initial_image_width
    rt = img_new_width/col
    image = cv2.resize(image, (img_new_width, int(row*rt)))
    lms = extractor.findFaceMesh(image.copy())
    
    if(lms is None):
        print("\nNo faces")
        return [], "No Faces detected"
    
    # --------------------------------------------------------------------------
    adjuster = FaceAdjuster(image.copy(), lms)
    _, succeed = adjuster.alignEyes()
    if not succeed:
        return [],[], adjuster.error
    _, succeed = adjuster.alignFace()
    if not succeed:
        return [],[], adjuster.error
    _, succeed = adjuster.faceCrop()
    if not succeed:
        return [],[], adjuster.error
    _, succeed = adjuster.alignFace()
    if not succeed:
        return [],[], adjuster.error
    finalImage, succeed = adjuster.fixImageSizeWithBorders()
    if not succeed:
        return [],[], adjuster.error

    nlms = adjuster.getLms()

    face_border = adjuster.find_face_border()
    return face_border,nlms, finalImage, None



def process_video(path = ""):
    camera = cv2.VideoCapture( str(path))
    vfps = (camera.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    name = path.split('/')
    name = name[len(name)-1].split('.')
    name = name[0]
    
    nprc_path = './vds/prc'+path.split("raw")[1]
    try:
        os.mkdir(nprc_path )
    except:
        print("Diretorio ja existe")
    path = nprc_path+  "/"
    name =  nprc_path +'/video.avi'
    vLength = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    #file = open(path+"data.pickle", 'rb')
    
    out = cv2.VideoWriter(name, fourcc,vfps,(final_image_size_width,final_image_size_height))
    
    positions_data = PositionsModule()
    landmarks_extractor = FaceMeshDetector()
    for i in tqdm(range (0,vLength+1)):
        
        ret, frame = camera.read()
        if ret: 
            frame_data = EyeDataModule(i)
            clear_image = frame.copy()
            try:
                face_border,lms, fimage, err = adjustFace(
                    clear_image, landmarks_extractor)
            except Exception as exception:
                err = type(exception).__name__
                print()
                print(err)
                continue


            # Identifica a iris e pega o raio
            eye_module = EyeModule(image=fimage,lms=lms)
            left_iris, right_iris = eye_module.detect_iris()


            left_pupil, right_pupil = eye_module.detect_pupil(left_iris, right_iris)
            if(left_pupil == None):
                left_iris = None
            if(right_pupil == None):
                right_iris = None
            
            #salva as posições
            frame_data.add_left_iris(left_iris)
            frame_data.add_right_iris(right_iris)
            frame_data.add_left_pupil(left_pupil)
            frame_data.add_right_pupil(right_pupil)


            #salva os dados totais
            positions_data.add_positions(frame_data)

            # desenha as coisas no rosto, descomenta o que não quiser mostrar
            #fimage = draw_face_box(fimage,face_border)
            fimage = draw_iris_circles(fimage,left_iris,right_iris)
            fimage = draw_pupil_circles(fimage,left_pupil,right_pupil)
            #fimage = draw_past_positions_iris_center(fimage,positions_data,20)
            #fimage = draw_face_mesh_points(image=fimage,lms=lms)
            
            cv2.imshow("adjusted",fimage)
            cv2.waitKey(1)

            #print("\nfimage shape ",fimage.shape)
            
            out.write(fimage)
    camera.release()
    cv2.destroyAllWindows()
    positions_data.save_data(path+"/positions.csv")

if __name__=="__main__":
    paths = []
    for root, dirs, files in os.walk('./vds/raw'):
        #print(root,dirs,files)
        nprc_path = './vds/prc'+root.split("raw")[1]
        #print(nprc_path)
        try:
            os.mkdir(nprc_path )
        except:
            None
        for file in files:
            print(root + '/' + file)
            process_video(path = root + '/' + file)
