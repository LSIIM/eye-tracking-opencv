
import sys
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
    'show_process': None,
    'draw_bb': None,
    'draw_iris': None,
    'draw_pupil': None,
    'draw_past_pos': None,
    'draw_mask_points': None,
    'show_warnings': None,
    'use_multicore': None,
    'overwrite': None,
    'draw_gaze': None,
    'path': None
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
    nprc_path = global_options['path'] + '/processed'
    try:
        os.mkdir(nprc_path)
    except:
        if (global_options['show_warnings']):
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
                if (global_options['draw_gaze']):
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
    for root, dirs, files in os.walk(global_options['path']):
        
        for file in files:
            if ((file.split(".")[1] == "avi" or file.split(".")[1] == "mp4") and (file.split(".")[0] == "record")):
                process_video(root + '/' + file)


def find_argument_by_option(option, arguments, default):
    for i in range(0, len(arguments)):
        if (arguments[i] == option):
            ans = arguments[i+1]
            # check if is valid
            if (ans == 's' or ans == 'n'):
                return ans == 's'
            else:
                print(f'Argumento {ans} invalido para a opção {option}. Abortando...')
                exit()
    return default == 's'

def get_path_argument(arguments, default):
    for i in range(0, len(arguments)):
        if (arguments[i] == '-path'):
            return arguments[i+1]
    return default

if __name__ == "__main__":

    # as options serão passadas por argumentos
    arguments = sys.argv[1:]

    global_options['show_process'] = find_argument_by_option(option = '-showprocess', arguments = arguments, default = 'n')
    global_options['draw_bb'] = find_argument_by_option(option = '-drawbb', arguments = arguments, default = 'n')
    global_options['draw_iris'] = find_argument_by_option(option = '-drawir', arguments = arguments, default = 'n')
    global_options['draw_pupil'] = find_argument_by_option(option = '-drawpu', arguments = arguments, default = 'n')
    global_options['draw_past_pos'] = find_argument_by_option(option = '-drawpp', arguments = arguments, default = 's')
    global_options['draw_mask_points'] = find_argument_by_option(option = '-drawmp', arguments = arguments, default = 'n')
    global_options['draw_gaze'] = find_argument_by_option(option = '-drawgz', arguments = arguments, default = 'n')
    global_options['show_warnings'] = find_argument_by_option(option = '-showwarn', arguments = arguments, default = 's')
    global_options['use_multicore'] = find_argument_by_option(option = '-multicore', arguments = arguments, default = 'n')
    
    if (global_options['use_multicore']):
        global_options['overwrite'] = find_argument_by_option(option = '-overwrite', arguments = arguments, default='s')
    else:
        global_options['overwrite'] = find_argument_by_option(option = '-overwrite', arguments = arguments, default='n')

    global_options['path'] = get_path_argument(arguments, default = './vds')
    
    if (not global_options['use_multicore']) :
        find_videos()
    else:
        processes = []
        ncpu = int(os.cpu_count()/2)
        print(f'Inicializando processamento multicore com {ncpu} nucleos')
        for i in range(ncpu):
            processes.append(Process(target=find_videos))

        for process in processes:
            process.start()

        for process in processes:
            process.join()
