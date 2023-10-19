
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
    original_path = path
    name_with_ext = path.split('/')
    name = name_with_ext[len(name_with_ext)-1].split('.')
    name_with_ext = name_with_ext[-1]
    name = name[0]
    # print(name)
    if (name == "auxiliary"):
        return
    nprc_path = global_options['path'] + '/processed' 

    if(global_options['show_warnings']):
        print(f'original path: {original_path}\nbase new path: {path}\nname: {name}\n')

    if(global_options['path'].split('/')[-1] !=  original_path.split('/')[-2]):
        # print(original_path.split(name_with_ext))
        # print(original_path.split(global_options['path']))
        diff = original_path.split(name_with_ext)[0].split(global_options['path'])[1]
        nprc_path += diff[:-1]

    if(global_options['show_warnings']):
        print(f'new path: {nprc_path}\n')   
    try:
        # mkdir que cria pasta e subpastas se necessário
        os.makedirs(nprc_path)
    except:
        if (global_options['show_warnings']):
            print("Diretorio ja existe")
        if (not global_options['overwrite']):
            return None, None
    path = nprc_path + "/"
    name = nprc_path + '/video.avi'
    if (global_options['show_warnings']):
        print(f'prc path: {path}\nprc name: {name}\n\n')

    return path, name
def process_video(path):
    vfps, vLength, camera, height, width = getVideoProperties(path)
    path, name = handle_directory(path)
    if(path is None):
        if(global_options['show_warnings']):
            print("Abortando processamento, pois o diretorio ja existe")
        return
    if(global_options['show_warnings']):
        print("\nProcessando: " + path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename=name, fourcc=fourcc, fps=vfps,
                          frameSize=(width, height), isColor=True)

    positions_data = PositionsModule()
    
    face_info = Face(logging = global_options['show_warnings'])
    face_not_found_counter = 0
    frame_not_found_counter = 0
    for i in tqdm(range(0, vLength)):
        ret, frame = camera.read()
        frame_data = FaceDataModule(i, height, width)
        
        if ret:
            original_frame = frame.copy()
            face_info.detect_face(frame)
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
            else:
                face_not_found_counter += 1

            # verifica se o frame é uma imagem valida
            if (frame is None):
                out.write(original_frame)
            else:
                out.write(frame)
            if (global_options['show_process']):
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            #print("\nframe shape ",frame.shape)
        else:
            frame_not_found_counter += 1
            if (global_options['show_warnings']):
                print("Frame nao encontrado, salvando com uma imagem preta")
            out.write(np.zeros((height, width, 3), np.uint8))

        # independente de ter encontrado a face ou nao, salva os dados    
        positions_data.add_positions(frame_data)


            
        
    out.release()
    camera.release()
    cv2.destroyAllWindows()
    positions_data.save_data(path+"/positions.csv")
    if (global_options['show_warnings']):
        print(f'Frames com faces nao encontradas: {face_not_found_counter}')
        print(f'Frames nao encontrados: {frame_not_found_counter}')

def verify_globals():
    # if it is running on multicore, the globals wont be shared, so we need to load it from the file
    if (global_options['path'] is None):
        print("Carregando opções do arquivo para o multicore")
        with open('options.txt', 'r') as f:
            for line in f:
                line = line.split()
                global_options[line[0]] = line[1] == 's'
        with open('path.txt', 'r') as f:
            global_options['path'] = f.readline()

def find_videos():
    verify_globals()
    for root, dirs, files in os.walk(global_options['path']): 
        root = root.replace('\\', '/')
        for file in files:
            if ((file.split(".")[1] == "avi" or file.split(".")[1] == "mp4") and (file.split(".")[0] == "record")):
                if(global_options['show_warnings']):
                    print(f'root: {root}\ndirs: {dirs}\nfile: {file}\n')
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
            # troca \ por / para evitar problemas com o windows
            return arguments[i+1].replace('\\', '/')
    return default

if __name__ == "__main__":

    # as options serão passadas por argumentos
    arguments = sys.argv[1:]

    # mostra os argumentos disponiveis e seus defauts e o que fazem (descrição breve)
    if (len(arguments) == 0):
        print('''
        Argumentos disponiveis:
        -showprocess s/n (default: n) -> mostra o video sendo processado    
        -drawbb s/n (default: n) -> desenha o bounding box da face
        -drawir s/n (default: n) -> desenha os circulos da iris
        -drawpu s/n (default: n) -> desenha os circulos da pupila
        -drawpp s/n (default: s) -> desenha as ultimas posicoes da pupila
        -drawmp s/n (default: n) -> desenha os pontos da malha da face
        -drawgz s/n (default: n) -> desenha o vetor de olhar
        -showwarn s/n (default: s) -> mostra avisos
        -multicore s/n (default: n) -> usa processamento multicore
        -overwrite s/n (default: s) -> sobrescreve os arquivos ja processados
        -path <path> (default: ./vds) -> caminho para a pasta com os videos
              ''')
        
        exit()

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
        global_options['overwrite'] = find_argument_by_option(option = '-overwrite', arguments = arguments, default='n')
    else:
        global_options['overwrite'] = find_argument_by_option(option = '-overwrite', arguments = arguments, default='s')

    global_options['path'] = get_path_argument(arguments, default = './vds')
    

    try:
        os.mkdir(global_options['path'] + '/processed')
    except:
        if (global_options['show_warnings']):
            print("Um processamento ja foi feito nessa pasta. Refazendo...")
    
     # save options on temp file
    with open('options.txt', 'w') as f:
        for key in global_options:
            if key !='path':
                f.write(f'{key} {"s" if global_options[key] else "n"}\n')
    with open('path.txt', 'w') as f:
        f.write(global_options['path'])

    # escreve uma copia em processed
    with open(global_options['path'] + '/processed/options.txt', 'w') as f:
        for key in global_options:
            if key !='path':
                f.write(f'{key} {"s" if global_options[key] else "n"}\n')
    with open(global_options['path'] + '/processed/path.txt', 'w') as f:
        f.write(global_options['path'])

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

    