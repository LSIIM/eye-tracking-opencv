import cv2
import mediapipe as mp


class FaceDetector():
    def __init__(self,
        staticImageMode=False,
        maxNumFaces=2,
        minDetectionConfidence=0.5,
        minTrackingConfidence=0.5):
        self.staticImageMode=bool(staticImageMode)
        self.maxNumFaces=int(maxNumFaces)
        self.minDetectionConfidence=float(minDetectionConfidence)
        self.minTrackingConfidence=float(minTrackingConfidence)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh

        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces =self.maxNumFaces ,min_detection_confidence =self.minDetectionConfidence, static_image_mode = self.staticImageMode,min_tracking_confidence = self.minTrackingConfidence )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)
    
    def findFaceMesh(self,img, draw = False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                face = []
                #print(faceLms)
                for i,lm in enumerate(faceLms.landmark):
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    if(draw):
                        cv2.putText(img,str(i),(x,y),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,0),1)
                    #print(i,x,y)
                    face.append([x,y,lm.z])
                faces.append(face)
        return img,faces
    def face_bottom(self,face):
        highest = None
        for lm in face:
            if highest == None:
                highest = lm[1]
                continue
            if lm[1] > highest:
                highest = lm[1]
        return highest
    def face_top(self,face):
        lowest = None
        for lm in face:
            if lowest == None:
                lowest = lm[1]
                continue
            if lm[1] < lowest:
                lowest = lm[1]
        return lowest
    def face_left(self,face):
        lowest = None
        for lm in face:
            if lowest == None:
                lowest = lm[0]
                continue
            if lm[0] < lowest:
                lowest = lm[0]
        return lowest
    def face_right(self,face):
        highest = None
        for lm in face:
            if highest == None:
                highest = lm[0]
                continue
            if lm[0] > highest:
                highest = lm[0]
        return highest
    
    #https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Left_Eye_shading.jpg
    def find_face_border(self,face):
        return self.face_top(face),self.face_left(face),self.face_bottom(face),self.face_right(face)
    
    def find_l_eye_border(self,face):
        #top,left,bottom,right
        return face[223][1],face[130][0],face[23][1],face[133][0]
    
    def find_r_eye_border(self,face):
        #top,left,bottom,right
        return face[257][1],face[463][0],face[253][1],face[263][0]