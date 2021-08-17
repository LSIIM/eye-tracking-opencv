import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
class iris_detection():
    def __init__(self, image=None, data_save = None):
        '''
        initialize the class and set the class attributes
        '''
        self._data = data_save
        self._img = image
        self._pupil = None
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

    def detect_iris_hist(self):
        gray = self.convert_to_gray_scale(self._img)

        
        o1,o2,o3,o4,o5 = self.apply_otsus(gray)
        blurred = cv2.bilateralFilter(o5,20,40,50)
        o1,o2,o3,o4,o5 = self.apply_otsus(blurred)
        '''cv2.imshow("OTSU 1", o1)
        cv2.imshow("OTSU 2", o2)
        cv2.imshow("OTSU 3", o3)
        cv2.imshow("OTSU 4", o4)
        cv2.imshow("OTSU 5", o5)'''
        cv2.imshow("blur",blurred)
        #ret, bin_img = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        ret, bin_img = cv2.threshold(blurred,int((2*blurred.mean())/2),255,cv2.THRESH_BINARY)


        kernel = np.ones((5,5),np.uint8)
        cv2.imshow("bin_iris",bin_img)
        
        '''bin_img = cv2.erode(bin_img,kernel,iterations = 3)
        cv2.imshow("Erosion",bin_img)'''
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("MORPH_OPEN IRIS",bin_img)
        bin_img = cv2.bilateralFilter(bin_img,50,100,50)
        bin_img = cv2.bilateralFilter(bin_img,50,100,50)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        
        ret, bin_img = cv2.threshold(bin_img,bin_img.min()+40,255,cv2.THRESH_BINARY)
        cv2.imshow("MORPH_CLOSE IRIS",bin_img)
        
        
        cv2.waitKey(1)


        row_count,col_count,media_c,media_r,desvio_c,desvio_r = self.hist_analisys(bin_img)
        
        if self._data != None:
            if type(self._data["left_eye"]["blurred"]) == np.ndarray:
                self._data["right_eye"]["blurred"] = blurred
                self._data["right_eye"]["binary"] = bin_img
                self._data["right_eye"]["histograms"]["rows"] = row_count
                self._data["right_eye"]["histograms"]["columns"] = col_count
            else:
                self._data["left_eye"]["blurred"] = blurred
                self._data["left_eye"]["binary"] = bin_img
                self._data["left_eye"]["histograms"]["rows"] = row_count
                self._data["left_eye"]["histograms"]["columns"] = col_count
        try:
            #print([int(media_c)],[int(media_r)], [[int(desvio_c),int(desvio_r)]])
            return [int(media_c)],[int(media_r)], [[int(desvio_c),int(desvio_r)]],self._data
        except:
            return [0],[0],[[0,0]],self._data
    def detect_pupil(self):
        # Read image
        
        gray = self.convert_to_gray_scale(self._img)
        o1,o2,o3,o4,o5 = self.apply_otsus(gray)
        
        o5 = cv2.bilateralFilter(o5,30,50,50)
        
        '''cv2.imshow("OTSU 1", o1)
        cv2.imshow("OTSU 2", o2)
        cv2.imshow("OTSU 3", o3)
        cv2.imshow("OTSU 4", o4)
        cv2.imshow("OTSU 5", o5)'''
        new_img = np.zeros(gray.shape, dtype=gray.dtype)
        min_val = o5.min()*1.2 + 30
        #print(new_img.shape)
        print(min_val)
        for i in range(0,new_img.shape[0]):
            for j in range(0,new_img.shape[1]):
                if(o5[i][j]>min_val):
                    new_img[i][j] = 255 
        new_img = np.array(new_img)
        
        cv2.imshow("d",new_img)
        kernel = np.ones((5,5),np.uint8)
        
        '''new_img = cv2.erode(new_img,kernel,iterations = 2)
        cv2.imshow("Erosion Pup",new_img)'''
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("MORPH_OPEN",new_img)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("MORPH_CLOSE",new_img)
        
        new_img = cv2.erode(new_img,kernel,iterations = 1)
        cv2.imshow("EROSION",new_img)
        
        
        row_count,col_count,media_c,media_r,desvio_c,desvio_r = self.hist_analisys(new_img)
        
        try:
            
            center = [int(media_c), int(media_r)]
            radius = int(desvio_c)
            return center,radius
        except:
            return [0,0],0
    def start_detection(self):
        if self._img is not None:
            #return self.detect_eye_features()
            return self.detect_iris_hist()
        else:
           #print('Image could not be loaded.')
            return None,None,None,self._data

