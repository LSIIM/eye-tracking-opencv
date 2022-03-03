

import cv2
import numpy as np
from numpy.core.fromnumeric import resize, shape
from scipy import ndimage
import math



from definitions import *
pi = 3.14159265359


class FaceAdjuster():
    def __init__(self, image, lms):
        self._img = image
        self._lms = lms
        self.error = ""

    def alignEyes(self):

        xR, yR = self._lms[362]
        xL, yL = self._lms[133]
        row, col = self._img.shape[:2]
        angle = math.atan2(
            (yR-yL), (xR-xL))
        rotated = ndimage.rotate(self._img, angle*180/pi, reshape=False)

        # print((yR-yL)/(xR-xL))  # tangente
        # print(math.atan2((yR-yL), (xR-xL)))  # radianos
        # print(math.atan2((yR-yL), (xR-xL))*180/pi)  # graus
        for i, lm in enumerate(self._lms):
            dy = ((row/2)-(lm[1]))
            dx = (-(col/2)+lm[0])

            #print("shift: ", angle)
            old_angle = math.atan2(
                dy, dx)

            #print("actual: ", old_angle)
            #print(self._lms[i], (dx, dy))

            r = math.sqrt(dy*dy + dx*dx)
            newposX = r * math.cos((old_angle+angle))
            newposY = r * math.sin((old_angle+angle))

            self._lms[i] = [int(newposX+(col/2)), int(-newposY + (row/2))]
        self._img = rotated
        return rotated, True

    def getImg(self):
        return self._img.copy()

    def getLms(self):
        return self._lms.copy()
    # The 2 paramiters lms are the 2 points of the face that will
    # be used to center the face and align it

    def alignFace(self):
        # deixa o rosto sempre na mesma posição
        rows, cols = self._img.shape[:2]
        pos = self._lms[10]
        distX = int(cols/2)-int(pos[0])
        distY = p10_height-int(pos[1])
        M = np.float32(
            [[1, 0, distX], [0, 1, distY]])
        dst = cv2.warpAffine(self._img, M, (cols, rows))

        for i, lm in enumerate(self._lms):
            self._lms[i] = [self._lms[i][0]+distX, self._lms[i][1]+distY]
        self._img = dst

        return dst, True

    def faceCrop(self):
        rows, cols = self._img.shape[:2]
        top, left, bottom, right = self.find_face_border()
        cent_img = []
        #print(top, left, bottom, right)
        if top < 0:
            top = 0
        if left < 0:
            left = 0

        for j in range(top, bottom):
            row = []
            if(j >= self._img.shape[0]):
                continue
            for i in range(left, right):
                if(i >= self._img.shape[1]):
                    continue
                row.append(self._img[j][i])
            cent_img.append(row)
        cent_img = np.array(cent_img)

        size = self._lms[152][1]-self._lms[10][1]
        propsize = face_height/size
        #print("Size: ", size)
        #print("Popsize: ", propsize)
        prop = cent_img.shape[0]/int(cent_img.shape[0]*propsize)
        cent_img = self._image_resize(
            cent_img, height=int(cent_img.shape[0]*propsize))

        #print("SHAPE: ", self._img.shape[:2])
        #print("PROP: ", prop)
        for i, lm in enumerate(self._lms):
            nx = int((self._lms[i][0]-left) / prop)
            ny = int((self._lms[i][1]-top)/prop)
            self._lms[i] = [nx, ny]
        self._img = cent_img
        return cent_img, True
    # private mathods

    def fixImageSizeWithBorders(self):
        height = final_image_size_height
        width = final_image_size_width
        row, col = self._img.shape[:2]
        hdif = height-row
        cdif = width-col
        left_cdif = int(cdif/2)
        right_cdif = int(cdif/2)
        if(2*int(cdif/2)+col>final_image_size_width):
            left_cdif-=1
        if(2*int(cdif/2)+col<final_image_size_width):
            left_cdif+=1
            
        try:
            border = cv2.copyMakeBorder(
                self._img,
                top=0,
                bottom=int(hdif),
                left=left_cdif,
                right=right_cdif,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
        except:
            print("Erro no tamanho: ", (row, col))
            for i in range(len(self._lms)):
                cv2.putText(self._img, ".", self._lms[i], cv2.FONT_HERSHEY_PLAIN,
                            0.8, (0, 255, 0), 1)
            cv2.imshow("Erro", self._img)
            cv2.waitKey(0)
            self.error = "Erro no tamanho: ", (row, col)
            return None, False
        for i, lm in enumerate(self._lms):
            nx = int((self._lms[i][0]+(cdif/2)))
            ny = self._lms[i][1]
            self._lms[i] = [nx, ny]
        self._img = border
        return border, True
# ------------------------------------------------------

    def _image_resize(self, img, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w, _) = img.shape

        if width is None and height is None:
            return img

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(img, dim, interpolation=inter)
        return resized

    def _getImage(self):
        return self._img

    def _face_bottom(self):
        highest = None
        for lm in self._lms:
            if highest == None:
                highest = lm[1]
                continue
            if lm[1] > highest:
                highest = lm[1]
        return highest

    def _face_top(self):
        lowest = None
        for lm in self._lms:
            if lowest == None:
                lowest = lm[1]
                continue
            if lm[1] < lowest:
                lowest = lm[1]
        return lowest

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

    # https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Left_Eye_shading.jpg
    def find_face_border(self):
        margin = face_margin
        top = self._face_top()-margin
        bottom = self._face_bottom()+margin
        left = self._face_left()-margin
        right = self._face_right()+margin

        if(top < 0):
            top = 0
        if(bottom > self._img.shape[0]):
            bottom = self._img.shape[0]
        if(left < 0):
            left = 0
        if(right > self._img.shape[1]):
            right = self._img.shape[1]
        return top, left, bottom, right

    def _find_l_eye_border(self):
        # top,left,bottom,right
        return self._lms[27][1], self._lms[130][0], self._lms[23][1], self._lms[133][0]

    def _find_r_eye_border(self):
        # top,left,bottom,right
        return self._lms[386][1], self._lms[362][0], self._lms[253][1], self._lms[263][0]
