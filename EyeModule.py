import cv2
import numpy as np
class iris_detection():
    def __init__(self, image):
        '''
        initialize the class and set the class attributes
        '''
        self._img = image
        self._pupil = None
    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)        
    def detect_iris(self):
        
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.bilateralFilter(gray,10,50,50)
        #cv2.imshow("Result b", blurred)
        #print(blurred.shape)
        minDist = 1
        param1 = 35 # 500
        param2 = 10 # 200 #smaller value-> more false circles
        minRadius = 5
        maxRadius = 300 #10

        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        x,y,r = 0,0,0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0][0]
            print(i[0])
            #print(i)
            x,y,r = i[0], i[1], i[2]
        return x,y,r
    def start_detection(self):

        if self._img is not None:
            return self.detect_iris()
        else:
            print('Image could not be loaded.')
            return None,None,None

