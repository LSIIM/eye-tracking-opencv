
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import pandas as pd

from multiprocessing import Process

from positions_module import EyeDataModule, PositionsModule
from face_adjustments_module import FaceAdjuster
from face_mesh_module import FaceMeshDetector
from eye_feature_detector_module import EyeModule
from definitions import *
from drawing_utils import *

def adjustFace(image, extractor,show_warnings):
    row, col = image.shape[:2]
    img_new_width = initial_image_width
    rt = img_new_width/col
    image = cv2.resize(image, (img_new_width, int(row*rt)))
    lms = extractor.findFaceMesh(image.copy())
    
    if(lms is None):
        if(show_warnings == 's'):
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



def process_video(path,show_process,draw_bb,draw_iris,draw_pupil,draw_past_pos,draw_mask_points,show_warnings,overwrite):
    camera = cv2.VideoCapture( str(path))
    vfps = (camera.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    name = path.split('/')
    name = name[len(name)-1].split('.')
    name = name[0]
    
    nprc_path = './vds/prc'+path.split("raw")[1].split(".")[0]
    try:
        os.mkdir(nprc_path )
    except:
        if(show_warnings == 's'):   
            print("Diretorio ja existe")
        if(overwrite == 'n'):
            return
    path = nprc_path+  "/"
    name =  nprc_path +'/video.avi'
    print("Processando: "+ path)
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
                    clear_image, landmarks_extractor,show_warnings)
            except Exception as exception:
                err = type(exception).__name__
                if(show_warnings == 's'):
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

            # desenha as coisas no rosto   

            if(draw_bb == 's'):
                fimage = draw_face_box(fimage,face_border)
            if(draw_iris == 's'):
                fimage = draw_iris_circles(fimage,left_iris,right_iris)
            if(draw_pupil == 's'):
                fimage = draw_pupil_circles(fimage,left_pupil,right_pupil)
            if(draw_past_pos == 's'):
                fimage = draw_past_positions_iris_center(fimage,positions_data,20)
            if(draw_mask_points == 's'):
                fimage = draw_face_mesh_points(image=fimage,lms=lms)


            
            if(show_process == 's'):   
                cv2.imshow("adjusted",fimage)
                cv2.waitKey(1)

            #print("\nfimage shape ",fimage.shape)
            
            out.write(fimage)
    camera.release()
    cv2.destroyAllWindows()
    positions_data.save_data(path+"/positions.csv")

def find_videos(show_process,draw_bb,draw_iris,draw_pupil,draw_past_pos,draw_mask_points,show_warnings,overwrite):
    for root, dirs, files in os.walk('./vds/raw'):
        nprc_path = './vds/prc'+root.split("raw")[1]
        try:
            os.mkdir(nprc_path )
        except:
            None
        for file in files:
            process_video(root + '/' + file,show_process,draw_bb,draw_iris,draw_pupil,draw_past_pos,draw_mask_points,show_warnings,overwrite)

if __name__=="__main__":

    print("As respostas a seguir serão usadas em todos os videos do processamento em lote!")
    show_process = str(input("(Isso pode implicar em perda de performance) Deseja mostrar o processo frame a frame? s/n ")).lower()
    draw_bb = str(input("Deseja desenhar uma borda ao redor do rosto? s/n ")).lower()
    draw_iris = str(input("Deseja desenhar os circulos da iris? s/n ")).lower()
    draw_pupil = str(input("Deseja desenhar os circulos das pupilas? s/n ")).lower()
    draw_past_pos = str(input("Deseja desenhar a linha com o rastreio das posições passadas? s/n ")).lower()
    draw_mask_points = str(input("Deseja desenhar os pontos da mascara no rosto? s/n ")).lower()
    show_warnings = str(input("Deseja que o mostre os avisos? s/n ")).lower()

    use_multicore = str(input("Deseja executar o programa em multicore? s/n ")).lower()
    if(use_multicore == 'n'):
        overwrite = str(input("Caso haja sobreposição dos processamentos, deseja sobrescrever o arquivo? s/n ")).lower()
        find_videos(show_process,draw_bb,draw_iris,draw_pupil,draw_past_pos,draw_mask_points,show_warnings,overwrite)
    else:
        overwrite = 'n'
        processes = []
        ncpu = int(os.cpu_count()/2)
        print(f'Inicializando processamento multicore com {ncpu} nucleos')
        for i in range(ncpu):
            processes.append(Process(target=find_videos,args=(show_process,draw_bb,draw_iris,draw_pupil,draw_past_pos,draw_mask_points,show_warnings,overwrite)))

        for process in processes:
            process.start()


        for process in processes:
            process.join()



    
