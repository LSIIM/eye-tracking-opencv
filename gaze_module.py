import cv2
import mediapipe as mp
import numpy as np
import time

class EyeGazeEstimator():
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        
        # 3D estimation for a human face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])

        # 3D model eye points
        self.eye_ball_center_left = np.array([[-29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.
        self.eye_ball_center_right = np.array([[29.05], [32.7], [-39.5]])  # the center of the right eyeball as a vector.

        # camera matrix estimation
        self.focal_length = self.frame_shape[1]
        self.camera_center = (self.frame_shape[1] / 2, self.frame_shape[0] / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def _relative(self, point):
        """Relative takes mediapipe points that is normalized to [-1, 1] and returns image points"""
        return (int(point[0]), int(point[1]))
    
    def _relativeT(self, point):
        """RelativeT takes mediapipe points that is normalized to [-1, 1] and returns image points at (x,y,0) format"""
        return (int(point[0]), int(point[1]), 0)

    def get_eye_gaze_vector(self, face_lms):
        """Returns the eye gaze vector of the face_lms [left_eye_vector, right_eye_vector]"""

        # 2D image points
        image_points_2D = np.array([
            self._relative(face_lms[4]),  # Nose tip
            self._relative(face_lms[152]),  # Chin
            self._relative(face_lms[263]),  # Left eye left corner
            self._relative(face_lms[33]),  # Right eye right corner
            self._relative(face_lms[287]),  # Left Mouth corner
            self._relative(face_lms[57])  # Right mouth corner
        ], dtype="double")

        # 2D.5 image points (it still 3d, nut the z is 0)
        image_points_2D5 = np.array([
            self._relativeT(face_lms[4]),  # Nose tip
            self._relativeT(face_lms[152]),  # Chin
            self._relativeT(face_lms[263]),  # Left eye, left corner
            self._relativeT(face_lms[33]),  # Right eye, right corner
            self._relativeT(face_lms[287]),  # Left Mouth corner
            self._relativeT(face_lms[57])  # Right mouth corner
        ], dtype="double")


        # Solve the PnP problem
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points_2D, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        # 2D Pupil location
        left_eye = self._relative(face_lms[473])
        right_eye = self._relative(face_lms[468])

        # Transformation between image point to world point
        _, transformation,_ = cv2.estimateAffine3D(image_points_2D5, self.model_points)
        if transformation is not None: # if estimateAffine3D succeeded
            # Project pupil image point into 3D world point
            left_pupil_world_cord = transformation @ np.array([[left_eye[0], left_eye[1], 0, 1]]).T

            right_pupil_world_cord = transformation @ np.array([[right_eye[0], right_eye[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary)
            gaze_distance_scale = 10
            left_S_vector = self.eye_ball_center_left + (left_pupil_world_cord - self.eye_ball_center_left) * gaze_distance_scale


            # Project a 3D gaze direction into the image plane
            (left_eye_pupil_2d, _) = cv2.projectPoints((int(left_S_vector[0]), int(left_S_vector[1]), int(left_S_vector[2])), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)


            # Project 3D head Pose into the image plane
            (head_pose, _) = cv2.projectPoints((int(left_pupil_world_cord[0]), int(left_pupil_world_cord[1]), int(40)), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)

            # correct gaze vector for the head pose
            left_eye_gaze = left_eye + (left_eye_pupil_2d[0][0] - left_eye) - (head_pose[0][0] - left_eye)

            right_eye_gaze = right_eye + (left_eye_pupil_2d[0][0] - right_eye) - (head_pose[0][0] - right_eye)

            return (left_eye_gaze, right_eye_gaze)
        else:
            return (None, None)


class HeadOrientationEstimator():
    def __init__(self, lms_3d, img_h, img_w):
        self.face_3d = []
        self.face_2d = []
        self.lms_3d = lms_3d
        self.img_h = img_h
        self.img_w = img_w
    
    def get_head_orientation_vector(self):
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

