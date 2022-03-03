import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
from definitions import *
class EyeModule():
    def __init__(self, image=None,  lms=None):
        '''
        initialize the class and set the class attributes
        '''
        self._img = image
        self._pupil = None
        self._lms = lms
        self._left_eye_img = None
        self._right_eye_img = None
    def convert_to_gray_scale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      

    
    def find_l_eye_border(self):
        #top,left,bottom,right
        return self._lms[223][1],self._lms[130][0],self._lms[23][1],self._lms[133][0]
    
    def find_r_eye_border(self):
        #top,left,bottom,right
        return self._lms[257][1],self._lms[463][0],self._lms[253][1],self._lms[263][0]
    

    def detect_iris(self):
        left_iris_points_pos = []
        right_iris_points_pos = []
        for lm in LEFT_IRIS:
            left_iris_points_pos.append(self._lms[lm])
        for lm in RIGHT_IRIS:
            right_iris_points_pos.append(self._lms[lm])
        

        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(np.array(right_iris_points_pos))
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(np.array(left_iris_points_pos))
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        return [[center_left, int(l_radius)],[center_right,int(r_radius)]]


        

    def crop_left_eye(self, img):
        topL,leftL,bottomL,rightL = self.find_l_eye_border()
        return img[topL:bottomL,leftL:rightL]
    def crop_right_eye(self, img):
        topR,leftR,bottomR,rightR = self.find_r_eye_border()
        return img[topR:bottomR,leftR:rightR]

    def apply_otsus(self,img):
        ret, o1 = cv2.threshold(img,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        ret, o2 = cv2.threshold(img,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
        ret, o3 = cv2.threshold(img,0,255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU )
        ret, o4 = cv2.threshold(img,0,255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU )
        ret, o5 = cv2.threshold(img,0,255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU )
        return [o1,o2,o3,o4,o5]

    def detect_pupil(self):
        # Read image
        
        gray = self.convert_to_gray_scale(self._img)
        
        left_eye_img = self.crop_left_eye(gray)
        right_eye_img = self.crop_right_eye(gray)
               
        

        #  Testes de filtos pra isolar a pupila
        left_eye_img = cv2.GaussianBlur(left_eye_img,(5,5),1)
        
        cv2.imshow("left_eye_img",left_eye_img)
        o1,o2,o3,o4,o5 = self.apply_otsus(left_eye_img)
        cv2.imshow("o1",o1)
        cv2.imshow("o2",o2)
        cv2.imshow("o3",o3)
        cv2.imshow("o4",o4)
        cv2.imshow("o5",o1)

        th2 = cv2.adaptiveThreshold(o4,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        blurred = cv2.bilateralFilter(o4,20,40,50)

        
        ret, bin_img = cv2.threshold(blurred,int((2*blurred.mean())/2),255,cv2.THRESH_BINARY)


        kernel = np.ones((5,5),np.uint8)
        
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        bin_img = cv2.bilateralFilter(bin_img,50,100,50)
        bin_img = cv2.bilateralFilter(bin_img,50,100,50)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        
        ret, bin_img = cv2.threshold(bin_img,bin_img.min()+40,255,cv2.THRESH_BINARY)
        cv2.imshow("MORPH_CLOSE",bin_img)
        
                # centro, raio
        return (100,100),20
       


