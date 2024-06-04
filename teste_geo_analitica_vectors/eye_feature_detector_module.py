from cProfile import label
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
from definitions import *
import math
import matplotlib.pyplot as plt


class EyeModule():
    def __init__(self, image=None,  lms=None):
        '''
        initialize the class and set the class attributes
        '''
        self._img = image
        self._pupil = None
        # dropa a informação z dos pontos e fica só com o x,y para cada lms
        self._lms = lms
        self._left_eye_img = None
        self._right_eye_img = None

    def convert_to_gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def find_l_eye_border(self):
        # top,left,bottom,right
        return self._lms[223][1], self._lms[130][0], self._lms[23][1], self._lms[133][0]

    def find_r_eye_border(self):
        # top,left,bottom,right
        return self._lms[257][1], self._lms[463][0], self._lms[253][1], self._lms[263][0]

    # funções auxiliares para detecção da iris

    def _face_left(self):
        lowest = None
        for lm in self._lms:
            if lowest == None:
                lowest = lm[0]
                continue
            if lm[0] < lowest:
                lowest = lm[0]
        return lowest

    def _face_right(self):
        highest = None
        for lm in self._lms:
            if highest == None:
                highest = lm[0]
                continue
            if lm[0] > highest:
                highest = lm[0]
        return highest

    def detect_iris(self):
        left_iris_points_pos = []
        right_iris_points_pos = []
        for lm in LEFT_IRIS:
            left_iris_points_pos.append(self._lms[lm])
        for lm in RIGHT_IRIS:
            right_iris_points_pos.append(self._lms[lm])

        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(
            np.array(right_iris_points_pos))
        if (r_cx < (self._face_left()-FACE_MARGIN) or r_cy < 0 or r_cx > (self._face_right()+FACE_MARGIN) or r_cy > self._img.shape[0]):
            riris = None
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        riris = [center_right, int(r_radius)]

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(
            np.array(left_iris_points_pos))

        if (l_cx < (self._face_left()-FACE_MARGIN) or l_cy < 0 or l_cx > (self._face_right()+FACE_MARGIN) or l_cy > self._img.shape[0]):
            liris = None
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        liris = [center_left, int(l_radius)]
        return [liris, riris]

    def crop_left_eye(self, img):
        return img[
            self._lms[LEFT_IRIS[1]][1]:self._lms[LEFT_IRIS[3]][1],
            self._lms[LEFT_IRIS[2]][0]:self._lms[LEFT_IRIS[0]][0]]

    def crop_right_eye(self, img):
        return img[
            self._lms[RIGHT_IRIS[1]][1]:self._lms[RIGHT_IRIS[3]][1],
            self._lms[RIGHT_IRIS[2]][0]:self._lms[RIGHT_IRIS[0]][0]]

    def apply_otsus(self, img):
        ret, o1 = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, o2 = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, o3 = cv2.threshold(
            img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        ret, o4 = cv2.threshold(
            img, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
        ret, o5 = cv2.threshold(
            img, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        return [o1, o2, o3, o4, o5]

    def hist_analisys(self, img):
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
            row_count[i] = (255 - row_count[i])
        for i in range(len(col_count)):
            col_count[i] = (255 - col_count[i])

        row_count = np.array(row_count)

        '''# plot on a graph the histograms
        
        fig = plt.figure()
        plt.plot(row_count, label="linha")
        plt.plot(col_count, label="coluna")
        plt.legend(loc="upper left")
        # show the graph

        plt.show()
        #plt.clf()'''

        row_count = row_count - row_count.min()
        media_r = 0
        desvio_r = 0
        media_c = 0
        desvio_c = 0
        if (len(row_count) > 0):
            for i, row in enumerate(row_count):
                media_r += row*i/row_count.sum()

            for i, row in enumerate(row_count):
                # if row>(0.45*row_count.max()):
                if row > 0:
                    desvio_r = media_r - i
                    break

            col_count = col_count - col_count.min()

            for i, col in enumerate(col_count):
                media_c += col*i/col_count.sum()

            for i, col in enumerate(col_count):
                if col > (col_count.max()*0.25):
                    # if col>0:
                    desvio_c = media_c - i
                    break

        if media_c is None or media_c.size <= 0 or media_c == NaN:
            media_c = 0
        if media_r is None or media_r.size <= 0 or media_r == NaN:
            media_r = 0
        if desvio_r is None or desvio_r.size <= 0 or desvio_r < 0:
            desvio_r = 0
        if desvio_c is None or desvio_c.size <= 0 or desvio_c < 0:
            desvio_c = 0
        return row_count, col_count, media_c, media_r, desvio_c, desvio_r

    def analyse_pupil(self, iris_img, iris_borders):
        # cv2.imshow("left_eye_img",left_eye_img)
        #  Testes de filtos pra isolar a pupila
        iris_img = cv2.GaussianBlur(iris_img, (5, 5), 1)

        # cv2.imshow("left_eye_img_blur",iris_img)
        o1, o2, o3, o4, o5 = self.apply_otsus(iris_img)
        b2 = cv2.GaussianBlur(o5, (5, 5), 1)

        th2 = cv2.adaptiveThreshold(b2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        blurred = cv2.bilateralFilter(b2, 20, 40, 50)

        # cv2.imshow("b2",b2)

        # cv2.imshow("th2",th2)

        # cv2.imshow("blur",blurred)
        #ret, bin_img = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        k = int(25*(len(b2.flatten())-1)/100)
        # print(len(b2.flatten()-1),k)
        idx = np.argpartition(b2.flatten(), k)
        # print(idx)
        # print(b2.flatten()[idx[:k]].mean())
        ret, bin_img = cv2.threshold(
            b2, b2.flatten()[idx[:k]].mean(), 255, cv2.THRESH_BINARY)

        #cv2.imshow("bin_img 1 ",bin_img)
        row_count, col_count, media_c, media_r, desvio_c, desvio_r = self.hist_analisys(
            bin_img)
        radius = int((desvio_c+desvio_r)/2)
        # print(int(media_c),int(media_r),radius)

        return (int(media_c) + self._lms[iris_borders[2]][0], int(media_r)+self._lms[iris_borders[1]][1]), radius

    def detect_pupil(self, left_iris, right_iris):
        # Read image

        gray = self.convert_to_gray_scale(self._img)

        left_eye_img = self.crop_left_eye(gray)
        right_eye_img = self.crop_right_eye(gray)

        if (left_iris):
            try:
                left_pupil = self.analyse_pupil(left_eye_img, LEFT_IRIS)
            except:
                left_pupil = None
        else:
            left_pupil = None
        if (right_iris):
            try:
                right_pupil = self.analyse_pupil(right_eye_img, RIGHT_IRIS)
            except:
                right_pupil = None
        else:
            right_pupil = None

        return [left_pupil, right_pupil]
