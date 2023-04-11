import cv2
import numpy as np
import pandas as pd
from definitions import *


class PositionsModule():
    def __init__(self):
        self._data = []

    def add_positions(self, face_data):
        self._data.append(face_data)

    def get_past_n_positions(self, n):
        left_eye = []
        right_eye = []
        for i in range(len(self._data)):
            eyeData = self._data[len(self._data)-(1+i)]
            if (i > n):
                break
            l_e = eyeData._left_iris
            r_e = eyeData._right_iris
            if (l_e == 0 or r_e == 0 or l_e == None or r_e == None):
                left_eye.append([None, None])
                right_eye.append([None, None])
            else:
                left_eye.append([eyeData._left_iris["x"], eyeData._left_iris['y']])
                right_eye.append(
                    [eyeData._right_iris["x"], eyeData._right_iris['y']])
        return left_eye, right_eye

    def save_data(self, path):
        df = pd.DataFrame()
        
        df_dict = {
            'frame': [ data._frame if data._frame is not None else "" for data in self._data],
            'height': [ data._height if data._height is not None else "" for data in self._data],
            'width': [ data._width if data._width is not None else "" for data in self._data],

            'left_iris_x': [ data._left_iris['x'] if data._left_iris is not None else "" for data in self._data],
            'left_iris_y': [ data._left_iris['y'] if data._left_iris is not None else "" for data in self._data],
            'left_iris_r': [ data._left_iris['r'] if data._left_iris is not None else "" for data in self._data],

            'right_iris_x': [ data._right_iris['x'] if data._right_iris is not None else "" for data in self._data],
            'right_iris_y': [ data._right_iris['y'] if data._right_iris is not None else "" for data in self._data],
            'right_iris_r': [ data._right_iris['r'] if data._right_iris is not None else "" for data in self._data],

            'left_pupil_x': [ data._left_pupil['x'] if data._left_pupil is not None else "" for data in self._data],
            'left_pupil_y': [ data._left_pupil['y'] if data._left_pupil is not None else "" for data in self._data],
            'left_pupil_r': [ data._left_pupil['r'] if data._left_pupil is not None else "" for data in self._data],

            'right_pupil_x': [ data._right_pupil['x'] if data._right_pupil is not None else "" for data in self._data],
            'right_pupil_y': [ data._right_pupil['y'] if data._right_pupil is not None else "" for data in self._data],
            'right_pupil_r': [ data._right_pupil['r'] if data._right_pupil is not None else "" for data in self._data],

            'gaze_x': [ data._gaze['x'] if data._gaze is not None else "" for data in self._data],
            'gaze_y': [ data._gaze['y'] if data._gaze is not None else "" for data in self._data],
            'gaze_z': [ data._gaze['z'] if data._gaze is not None else "" for data in self._data],
        }

       
        df = pd.DataFrame(df_dict)
        df.to_csv(path)


class FaceDataModule():
    def __init__(self, frame,height, width):
        self._left_iris = None
        self._right_iris = None
        self._left_pupil = None
        self._right_pupil = None
        self._gaze = None
        self._frame = frame
        self._height = height
        self._width = width
    def print_data(self):
        print(f'\nFrame: {self._frame}')
        print("\nLeft Iris:")
        print(self._left_iris)
        print("\nRight Iris:")
        print(self._right_iris)
        print("\nLeft Pupil:")
        print(self._left_pupil)
        print("\nRight Pupil:")
        print(self._right_pupil)
        print("\n-------------------------------------")

    def add_gaze_data(self, data):
        if(data is not None):
            x,y,z = data
            self._gaze = {
                "x": x,
                "y": y,
                "z": z
            }
        else:
            self._gaze = {
                "x": None,
                "y": None,
                "z": None
            }

    def add_position_data(self, data, key):
        if(data is not None):
            (x, y), r = data

            # get the self by the key
            self.__dict__["_" + key] = {
                "x": x,
                "y": y,
                "r": r
            }
        
        else:
            self.__dict__["_" + key] = {
                "x": None,
                "y": None,
                "r": None
            }

    