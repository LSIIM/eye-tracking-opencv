import cv2
import mediapipe as mp
import numpy as np

from definitions import *

class FaceMeshDetector():
    def __init__(self,
                 staticImageMode=False,  # False se analisar video
                 maxNumFaces=1,
                 minDetectionConfidence=0.75,
                 minTrackingConfidence=0.65): # esse aqui é ignorado se static for True
 
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
        img_h, img_w = image.shape[:2]
        if results.multi_face_landmarks:
            # mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3d = np.array([np.multiply([p.x, p.y, p.z], [img_w, img_h, 1]).astype(float) for p in results.multi_face_landmarks[0].landmark])
            
            return mesh_points_3d
        else:
            return None
