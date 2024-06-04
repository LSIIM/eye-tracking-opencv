import cv2
import mediapipe as mp
import numpy as np
import time

class EyeGazeEstimator():
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        
        

class HeadOrientationEstimator():
    def __init__(self, lms_3d, img_h, img_w):
        self.face_3d = []
        self.face_2d = []
        self.lms_3d = lms_3d
        self.img_h = img_h
        self.img_w = img_w
    
    