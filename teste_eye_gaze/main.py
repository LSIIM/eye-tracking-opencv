import mediapipe as mp
import cv2
import gaze

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

# camera stream:
cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)
# open kid.avi as the cap   
# cap = cv2.VideoCapture('kid.avi')
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:  # no frame input
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

        if results.multi_face_landmarks:
            gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation

            # for point in results.multi_face_landmarks[0].landmark:
            #     x = int(point.x * image.shape[1])
            #     y = int(point.y * image.shape[0])
            #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
            

        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()
