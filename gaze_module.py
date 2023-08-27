import cv2
import mediapipe as mp
import numpy as np
import time

class GazeEstimator():
    def __init__(self, lms_3d, img_h, img_w):
        self.face_3d = []
        self.face_2d = []
        self.lms_3d = lms_3d
        self.img_h = img_h
        self.img_w = img_w
    
    def get_gaze_vector(self):
        for idx, lm in enumerate(self.lms_3d):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm[0] , lm[1] )
                    nose_3d = (lm[0] , lm[1] , lm[2]*3000 )

                x, y = int(lm[0] ), int(lm[1] )

                # Get the 2D Coordinates
                self.face_2d.append([x, y])

                # Get the 3D Coordinates
                self.face_3d.append([x, y, lm[2]])
        # convert to np
        self.face_2d = np.array(self.face_2d, dtype=np.float64)
        self.face_3d = np.array(self.face_3d, dtype=np.float64)

        # the cara matrix
        focal_length = 1 * self.img_w

        cam_matrix = np.array([
            [focal_length, 0, self.img_h/2],
            [0, focal_length, self.img_w/2],
            [0, 0, 1]
        ])

        # the distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.face_3d, self.face_2d, cam_matrix, dist_matrix)
        
        # get the rotation matrix
        rot_mat, jac = cv2.Rodrigues(rot_vec)

        # get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_mat)

        # get the rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        return (x, y, z), nose_2d

