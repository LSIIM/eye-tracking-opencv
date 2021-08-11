import cv2
import numpy as np
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
    def detect_eye_features(self):
        
        gray = self.convert_to_gray_scale(self._img)

        
        o1,o2,o3,o4,o5 = self.apply_otsus(gray)
        #cv2.imshow("OTSU 1", o1)
        #cv2.imshow("OTSU 2", o2)
        #cv2.imshow("OTSU 3", o3)
        #cv2.imshow("OTSU 4", o4)
        #cv2.imshow("OTSU 5", o5)
        blured = cv2.bilateralFilter(o5,10,50,50)
        #cv2.imshow("bin_img", bin_img)


        #print(bin_img.shape)
        minDist = 1
        param1 = 25 # 500
        param2 = 20 # 200 #smaller value-> more false circles
        minRadius = 1
        maxRadius = 300 #10

        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        x,y,r = [0],[0],[[0,0]]
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0][0]
            x,y,r = [i[0]], [i[1]], [[i[2],i[2]]]
            if len(circles[0])>1:
                #print(circles[0],len(circles[0]))
                #print(circles[0][0])
                if circles[0][1].all is not circles[0][0]:
                    j = circles[0][1]
                    x.append(j[0])
                    y.append(j[1])
                    r.append([j[2],j[2]])
            #print(i[0])
            #print(i)
        return x,y,r
    
    def detect_iris_hist(self):
        gray = self.convert_to_gray_scale(self._img)

        
        o1,o2,o3,o4,o5 = self.apply_otsus(gray)
        #cv2.imshow("OTSU 1", o1)
        #cv2.imshow("OTSU 2", o2)
        #cv2.imshow("OTSU 3", o3)
        #cv2.imshow("OTSU 4", o4)
        #cv2.imshow("OTSU 5", o5)
        blurred = cv2.bilateralFilter(o5,10,50,50)
        
        #ret, bin_img = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        ret, bin_img = cv2.threshold(blurred,int((blurred.mean() + blurred.max())/2),255,cv2.THRESH_BINARY)
        #cv2.imshow("Blurred", blurred)
        #cv2.imshow("Binary", bin_img)

        col_count = np.zeros((bin_img.shape[1]))
        row_count = []
        for i in range(bin_img.shape[0]):
            aux_row = 0
            aux_col = 0
            for j in range(bin_img.shape[1]):
                aux_row = aux_row + bin_img[i][j]/bin_img.shape[1]
                col_count[j] = col_count[j] + bin_img[i][j]/bin_img.shape[0]
            row_count.append(aux_row)
        #cv2.imshow("bin_img", bin_img)
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
                if col>(0.45*col_count.max()):
                #if col>0:
                    desvio_c = media_c - i
                    break
        if media_c is None or media_c == [] :
            media_c = 0
        if media_r is None or media_r == [] :
            media_r = 0
        if desvio_r is None or desvio_r == [] or desvio_r<0 :
            desvio_r = 0
        if desvio_c is None or desvio_c == [] or desvio_c<0 :
            desvio_c = 0
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
        
        new_img = np.zeros(gray.shape, dtype=gray.dtype)
        min_val = self._img.min() + 40
        #print(new_img.shape)
        print(min_val)
        for i in range(0,new_img.shape[0]):
            for j in range(0,new_img.shape[1]):
                if(gray[i][j]>min_val):
                    new_img[i][j] = 255 
        new_img = np.array(new_img)
        
        cv2.imshow("d",new_img)
        kernel = np.ones((5,5),np.uint8)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("MORPH_OPEN",new_img)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("MORPH_CLOSE",new_img)
        
        contours,hierarchy = cv2.findContours(new_img, 1, 2)
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = [int(x),int(y)]
            radius = int(radius)
            cv2.circle(self._img,center,radius,(0,255,255),2)
            cv2.imshow("dd",self._img)
            cv2.waitKey(2)
        if(len(contours) == 0):
            return (0,0),0
        
        if len(contours)>1:
            cnt = contours[1]
        else:
            cnt = contours[0]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = [int(x),int(y)]
        radius = int(radius)
        return center,radius
    def start_detection(self):
        if self._img is not None:
            #return self.detect_eye_features()
            return self.detect_iris_hist()
        else:
           #print('Image could not be loaded.')
            return None,None,None,self._data

