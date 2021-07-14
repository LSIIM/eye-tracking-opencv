#https://google.github.io/mediapipe/solutions/face_detection

import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
aux = 1
l_eye_m = (0,0)
l_eye_m_aux = (0,0)

r_eye_m = (0,0)
r_eye_m_aux = (0,0)

eye_med_freq = 5
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        detection = results.detections[0]
        #mp_drawing.draw_detection(image, detection)
        points = detection.location_data.relative_keypoints
        l_eye_m_aux = (l_eye_m_aux[0] +int(points[0].x*image.shape[1]/eye_med_freq),l_eye_m_aux[1] + int(points[0].y*image.shape[0]/eye_med_freq) )
        if l_eye_m == (0,0):
            l_eye_m = (int(points[0].x*image.shape[1]),int(points[0].y*image.shape[0]))
        
        r_eye_m_aux = (r_eye_m_aux[0] +int(points[1].x*image.shape[1]/eye_med_freq),r_eye_m_aux[1] + int(points[1].y*image.shape[0]/eye_med_freq) )
        if r_eye_m == (0,0):
            r_eye_m = (int(points[1].x*image.shape[1]),int(points[1].y*image.shape[0]))

    if aux == eye_med_freq:
        aux = 0
        l_eye_m = l_eye_m_aux
        l_eye_m_aux = (0,0)
        r_eye_m = r_eye_m_aux
        r_eye_m_aux = (0,0)
    cv2.putText(image,"+",(l_eye_m),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    cv2.putText(image,"+",(r_eye_m),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    aux+=1
cap.release()