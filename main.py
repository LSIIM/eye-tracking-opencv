
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import pandas as pd

from multiprocessing import Process

from positions_module import FaceDataModule, PositionsModule
from definitions import *
from drawing_utils import *

from Face import Face

global_options = {
    'save_video': None,
    'show_process': None,
    'draw_bb': None,
    'draw_iris': None,
    'draw_pupil': None,
    'draw_past_pos': None,
    'draw_mask_points': None,
    'show_warnings': None,
    'use_multicore': None,
    'overwrite': None,
    'show_gaze': None
}

def getVideoProperties(path):
    camera = cv2.VideoCapture(str(path))
    vfps = (camera.get(cv2.CAP_PROP_FPS))
    vLength = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    return vfps, vLength, camera, height, width

def handle_directory(path):
    name = path.split('/')
    name = name[len(name)-1].split('.')
    name = name[0]
    # print(name)
    if (name == "auxiliary"):
        return
    nprc_path = './vds/prc'+path.split("raw")[1].split(".")[0]
    try:
        os.mkdir(nprc_path)
    except:
        if (show_warnings == 's'):
            print("Diretorio ja existe")
        if (not global_options['overwrite']):
            return None, None
    path = nprc_path + "/"
    name = nprc_path + '/video.avi'

    return path, name
def process_video(path):
    

    vfps, vLength, camera, height, width = getVideoProperties(path)
    path, name = handle_directory(path)
    if(path is None):
        if(global_options['show_warnings']):
            print("Abortando processamento, pois o diretorio ja existe")
        return
    print("\nProcessando: " + path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(name, fourcc, vfps,
                          (width, height))

    positions_data = PositionsModule()
    for i in tqdm(range(0, vLength+1)):
        ret, frame = camera.read()
        frame_data = FaceDataModule(i, height, width)
        
        if ret:
            
            
            face_info = Face(frame, logging = global_options['show_warnings'])
            face_info.detect_face()
            if(face_info.lms_3d is not None):
                
                face_info.init_eye_module()
                face_info.detect_iris()
                face_info.detect_pupil()

                face_info.detect_gaze()

                # salva os dados da face
                face_data_dict = face_info.get_position_data_as_dict()
                # para cada chave do dicionario, salva o valor no dicionario de dados
                for key in face_data_dict:
                    frame_data.add_position_data(face_data_dict[key],key)
                
                frame_data.add_gaze_data(face_info.gaze_vector)

                # desenha os dados da face
                if (global_options['draw_bb']):
                    face_info.find_face_border()
                    frame = draw_face_box(frame, face_info.face_border)
                if (global_options['draw_iris']):
                    frame = draw_iris_circles(frame, face_info.left_iris, face_info.right_iris)
                if (global_options['draw_pupil']):
                    frame = draw_pupil_circles(frame, face_info.left_pupil, face_info.right_pupil)
                if (global_options['draw_past_pos']):
                    frame = draw_past_positions_iris_center(
                        frame, positions_data, 100)
                if (global_options['draw_mask_points']):
                    frame = draw_face_mesh_points(image=frame, lms=face_info.lms_2d)
                if (global_options['show_gaze']):
                    frame = draw_gaze(frame, face_info.gaze_vector, face_info.nose_2d)

            if (global_options['show_process']):
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            #print("\nframe shape ",frame.shape)

        positions_data.add_positions(frame_data)
        out.write(frame)
        
    out.release()
    camera.release()
    cv2.destroyAllWindows()
    positions_data.save_data(path+"/positions.csv")


def find_videos():
    for root, dirs, files in os.walk('./vds/raw'):
        nprc_path = './vds/prc'+root.split("raw")[1]
        try:
            os.mkdir(nprc_path)
        except:
            None
        for file in files:
            if (file.split(".")[1] == "avi" or file.split(".")[1] == "mp4"):
                process_video(root + '/' + file)


if __name__ == "__main__":


    print("As respostas a seguir serão usadas em todos os videos do processamento em lote!")




    show_process = str(input("(Isso pode implicar em perda de performance) Deseja mostrar o processo frame a frame? s/n ")).lower()
    global_options['show_process'] = show_process == 's'
    
    draw_bb = str(input("Deseja desenhar uma borda ao redor do rosto? s/n ")).lower()
    global_options['draw_bb'] = draw_bb == 's'

    draw_iris = str(input("Deseja desenhar os circulos da iris? s/n ")).lower()
    global_options['draw_iris'] = draw_iris == 's'
    
    draw_pupil = str(input("Deseja desenhar os circulos das pupilas? s/n ")).lower()
    global_options['draw_pupil'] = draw_pupil == 's'
    
    draw_past_pos = str(input("Deseja desenhar a linha com o rastreio das posições passadas? s/n ")).lower()
    global_options['draw_past_pos'] = draw_past_pos == 's'
    
    draw_mask_points = str(input("Deseja desenhar os pontos da mascara no rosto? s/n ")).lower()
    global_options['draw_mask_points'] = draw_mask_points == 's'

    show_gaze = str(input("Deseja mostrar o ponto de foco do olhar? s/n ")).lower()
    global_options['show_gaze'] = show_gaze == 's'
    
    show_warnings = str(input("Deseja que o mostre os avisos? s/n ")).lower() 
    global_options['show_warnings'] = show_warnings == 's'


    use_multicore = str(input("Deseja executar o programa em multicore? s/n ")).lower()
    global_options['use_multicore'] = use_multicore == 's'
    
    if (not global_options['use_multicore']) :
        overwrite = str(input("Caso haja sobreposição dos processamentos, deseja sobrescrever o arquivo? s/n ")).lower()
        global_options['overwrite'] = overwrite == 's'
        find_videos()
    else:
        global_options['overwrite'] = False

        processes = []
        ncpu = int(os.cpu_count()/2)
        print(f'Inicializando processamento multicore com {ncpu} nucleos')
        for i in range(ncpu):
            processes.append(Process(target=find_videos))

        for process in processes:
            process.start()

        for process in processes:
            process.join()
