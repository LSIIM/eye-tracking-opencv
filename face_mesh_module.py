import cv2
import mediapipe as mp
import numpy as np

from definitions import *

class FaceMeshDetector():
    def __init__(self,
                 staticImageMode=True,  # False se analisar video
                 maxNumFaces=1,
                 minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):

        self._staticImageMode = bool(staticImageMode)
        self._maxNumFaces = int(maxNumFaces)
        self._minDetectionConfidence = float(minDetectionConfidence)
        self._minTrackingConfidence = float(minTrackingConfidence)

        self._mpDraw = mp.solutions.drawing_utils
        self._mpFaceMesh = mp.solutions.face_mesh

        self._faceMesh = self._mpFaceMesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=self._maxNumFaces,
                min_detection_confidence=self._minDetectionConfidence,
                static_image_mode=self._staticImageMode,
                min_tracking_confidence=self._minTrackingConfidence)

        self._drawSpec = self._mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, image):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._faceMesh.process(imgRGB)
        lms = []
        # tenho que mudar isso pra np array
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                face = []
                for lm in faceLms.landmark:
                    ih, iw, ic = image.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append((x, y))
                lms.append(face)

            return np.array(lms[0])
        else:
            return None
