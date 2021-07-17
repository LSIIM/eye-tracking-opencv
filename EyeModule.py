import cv2
import numpy as np
class iris_detection():
    def __init__(self, image):
        '''
        initialize the class and set the class attributes
        '''
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
        blurred = cv2.bilateralFilter(o5,10,50,50)
        #cv2.imshow("Blurred", blurred)


        #print(blurred.shape)
        minDist = 1
        param1 = 25 # 500
        param2 = 20 # 200 #smaller value-> more false circles
        minRadius = 1
        maxRadius = 300 #10

        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        x,y,r = [1],[1],[[1,1]]
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
        #cv2.imshow("Blurred", blurred)

        col_count = np.zeros((blurred.shape[1]))
        row_count = []
        for i in range(blurred.shape[0]):
            aux_row = 0
            aux_col = 0
            for j in range(blurred.shape[1]):
                aux_row = aux_row + blurred[i][j]/blurred.shape[1]
                col_count[j] = col_count[j] + blurred[i][j]/blurred.shape[0]
            row_count.append(aux_row)
        #cv2.imshow("Blurred", blurred)
        for i in range(len(row_count)):
            row_count[i] =( 255 -row_count[i])
        for i in range(len(col_count)):
            col_count[i] = (255 -col_count[i])
        
        row_count = np.array(row_count)
        
        row_count = row_count - row_count.min()
        media_r = 0
        for i,row in enumerate(row_count):
            media_r+= row*i/row_count.sum()
        desvio_r = 0
        for i,row in enumerate(row_count):
            #if row>(0.45*row_count.max()):
            if row>0:
                desvio_r = media_r - i
                break
        
        col_count = col_count - col_count.min()
        media_c = 0
        for i,col in enumerate(col_count):
            media_c+= col*i/col_count.sum()
        desvio_c = 0
        for i,col in enumerate(col_count):
            if col>(0.45*col_count.max()):
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

        try:
            #print([int(media_c)],[int(media_r)], [[int(desvio_c),int(desvio_r)]])
            return [int(media_c)],[int(media_r)], [[int(desvio_c),int(desvio_r)]]
        except:
            return [1],[1],[[1,1]]

    def start_detection(self):
        if self._img is not None:
            #return self.detect_eye_features()
            return self.detect_iris_hist()
        else:
           #print('Image could not be loaded.')
            return None,None,None

