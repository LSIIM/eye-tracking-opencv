import cv2
import numpy as np
from definitions import *
class PositionsModule():
    def __init__(self):
        self._data = []
    
    def add_positions(self,eyeData):
        self._data.append(eyeData)

    def get_past_n_positions(self,n):
        left_eye = []
        right_eye = []
        for i in range(len(self._data)):
            eyeData = self._data[len(self._data)-(1+i)]
            if(i>n):
                break
            left_eye.append([eyeData._left_iris["x"],eyeData._left_iris['y']])
            right_eye.append([eyeData._right_iris["x"],eyeData._right_iris['y']])
        return left_eye,right_eye
    def save_data(self,path):
        return True


class EyeDataModule():
    def __init__(self,frame):
        self._left_iris = {}
        self._right_iris = {}
        self._left_pupil = {}
        self._right_pupil = {}
        self._frame = frame

    def add_left_iris(self, left_iris):
        (x,y),r = left_iris
        
        self._left_iris = {
                    "x": x,
                    "y": y,
                    "r": r
                }
    def add_right_iris(self, right_iris):
        (x,y),r = right_iris
        self._right_iris = {
                    "x": x,
                    "y": y,
                    "r": r
                }
    def add_left_pupil(self, left_pupil):
        (x,y),r = left_pupil
        
        self._left_pupil = {
                    "x": x,
                    "y": y,
                    "r": r
                }
    def add_right_pupil(self, right_pupil):
        (x,y),r = right_pupil
        
        self._right_pupil = {
                    "x": x,
                    "y": y,
                    "r": r
                }

        