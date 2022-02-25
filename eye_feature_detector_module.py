import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
from definitions import *
class EyeModule():
    def __init__(self, image=None, data_save = None, lms=None):
        '''
        initialize the class and set the class attributes
        '''
        self._data = data_save
        self._img = image
        self._pupil = None
        self._lms = lms
    def convert_to_gray_scale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      

    def apply_otsus(self,img):
        ret, o1 = cv2.threshold(img,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        ret, o2 = cv2.threshold(img,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
        ret, o3 = cv2.threshold(img,0,255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU )
        ret, o4 = cv2.threshold(img,0,255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU )
        ret, o5 = cv2.threshold(img,0,255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU )
        return [o1,o2,o3,o4,o5]
    
    def hist_analisys(self,img):
        col_count = np.zeros((img.shape[1]))
        row_count = []
        for i in range(img.shape[0]):
            aux_row = 0
            aux_col = 0
            for j in range(img.shape[1]):
                aux_row = aux_row + img[i][j]/img.shape[1]
                col_count[j] = col_count[j] + img[i][j]/img.shape[0]
            row_count.append(aux_row)
        #cv2.imshow("img", img)
        for i in range(len(row_count)):
            row_count[i] =( 255 -row_count[i])
        for i in range(len(col_count)):
            col_count[i] = (255 -col_count[i])
        
        row_count = np.array(row_count)
        
        row_count = row_count - row_count.min()
        media_r = 0
        desvio_r = 0
        media_c = 0
        desvio_c = 0
        if(len(row_count)>0):
            for i,row in enumerate(row_count):
                media_r+= row*i/row_count.sum()

            for i,row in enumerate(row_count):
                #if row>(0.45*row_count.max()):
                if row>0:
                    desvio_r = media_r - i
                    break
            
            col_count = col_count - col_count.min()
            
            for i,col in enumerate(col_count):
                media_c+= col*i/col_count.sum()

            for i,col in enumerate(col_count):
                if col>(col_count.max()*0.25):
                #if col>0:
                    desvio_c = media_c - i
                    break
        
        if media_c is None or media_c == [] or media_c == NaN :
            media_c = 0
        if media_r is None or media_r == [] or media_r == NaN:
            media_r = 0
        if desvio_r is None or desvio_r == [] or desvio_r<0 :
            desvio_r = 0
        if desvio_c is None or desvio_c == [] or desvio_c<0 :
            desvio_c = 0
        return row_count,col_count,media_c,media_r,desvio_c,desvio_r

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
        
    def detect_pupil(self):
        # Read image
        
        gray = self.convert_to_gray_scale(self._img)
        o1,o2,o3,o4,o5 = self.apply_otsus(gray)
        
        o5 = cv2.bilateralFilter(o5,30,50,50)
        
        o5 = cv2.medianBlur(o5,1)
        
        '''cv2.imshow("OTSU 1", o1)
        cv2.imshow("OTSU 2", o2)
        cv2.imshow("OTSU 3", o3)
        cv2.imshow("OTSU 4", o4)
        cv2.imshow("OTSU 5", o5)'''
        new_img = np.zeros(gray.shape, dtype=gray.dtype)
        min_val = o5.min()*1.2 + 30
        #print(new_img.shape)
        #(min_val)
        for i in range(0,new_img.shape[0]):
            for j in range(0,new_img.shape[1]):
                if(o5[i][j]>min_val):
                    new_img[i][j] = 255 
        new_img = np.array(new_img)
        
        #cv2.imshow("d",new_img)
        kernel = np.ones((5,5),np.uint8)
        
        '''new_img = cv2.erode(new_img,kernel,iterations = 2)
        cv2.imshow("Erosion Pup",new_img)'''
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("MORPH_OPEN",new_img)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow("MORPH_CLOSE",new_img)
        
        new_img = cv2.erode(new_img,kernel,iterations = 1)
        #cv2.imshow("EROSION",new_img)
        
        
        row_count,col_count,media_c,media_r,desvio_c,desvio_r = self.hist_analisys(new_img)
        
        try:
            
            center = [int(media_c), int(media_r)]
            radius = int(desvio_c)
            return center,radius
        except:
            return [0,0],0


