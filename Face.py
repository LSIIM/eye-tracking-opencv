import cv2
import mediapipe as mp
import numpy as np


from face_mesh_module import FaceMeshDetector
from eye_feature_detector_module import EyeModule
from definitions import *


class Face():
    def __init__(self, image, logging = False):
        self.image = image.copy()
        self._face_mesh_detector = FaceMeshDetector()
        self._eye_module = None
        self.logging = logging
        self.lms = None
        self.left_iris = None
        self.right_iris = None
        self.left_pupil = None
        self.right_pupil = None
        self.face_border = None

    def get_data_as_dict(self):
        return {
            'left_iris': self.left_iris,
            'right_iris': self.right_iris,
            'left_pupil': self.left_pupil,
            'right_pupil': self.right_pupil,
            # 'lms': self.lms
        }
    def init_eye_module(self):
        self._eye_module = EyeModule(self.image, self.lms)
    def detect_face(self):
        self.lms = self._face_mesh_detector.findFaceMesh(self.image)
        if self.lms is None and self.logging:
            print("Face nao encontrada")
    
    def detect_iris(self):
        left_iris, right_iris = self._eye_module.detect_iris()
        self.left_iris = left_iris
        self.right_iris = right_iris
    
    def detect_pupil(self):
        left_pupil, right_pupil = self._eye_module.detect_pupil(
            self.left_iris,
            self.right_iris)
        
        if (left_pupil == None):
            self.left_iris = None
        if (right_pupil == None):
            self.right_iris = None

        self.left_pupil = left_pupil
        self.right_pupil = right_pupil

    # https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Left_Eye_shading.jpg
    def find_face_border(self):
        margin = FACE_MARGIN
        top = self._face_top()-margin
        bottom = self._face_bottom()+margin
        left = self._face_left()-margin
        right = self._face_right()+margin

        if(top < 0):
            top = 0
        if(bottom > self.image.shape[0]):
            bottom = self.image.shape[0]
        if(left < 0):
            left = 0
        if(right > self.image.shape[1]):
            right = self.image.shape[1]
        self.face_border =  top, left, bottom, right

    def _find_l_eye_border(self):
        # top,left,bottom,right
        return self.lms[27][1], self.lms[130][0], self.lms[23][1], self.lms[133][0]

    def _find_r_eye_border(self):
        # top,left,bottom,right
        return self.lms[386][1], self.lms[362][0], self.lms[253][1], self.lms[263][0]

    #  --------------------------------------------
    # private methods
    def _face_bottom(self):
        highest = None
        for lm in self.lms:
            if highest == None:
                highest = lm[1]
                continue
            if lm[1] > highest:
                highest = lm[1]
        return highest

    def _face_top(self):
        lowest = None
        for lm in self.lms:
            if lowest == None:
                lowest = lm[1]
                continue
            if lm[1] < lowest:
                lowest = lm[1]
        return lowest

    def _face_left(self):
        lowest = None
        for lm in self.lms:
            if lowest == None:
                lowest = lm[0]
                continue
            if lm[0] < lowest:
                lowest = lm[0]
        return lowest

    def _face_right(self):
        highest = None
        for lm in self.lms:
            if highest == None:
                highest = lm[0]
                continue
            if lm[0] > highest:
                highest = lm[0]
        return highest