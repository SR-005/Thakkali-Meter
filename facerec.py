import cv2
import mediapipe as mp
import math

feed = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
feed.set(3,640) 
feed.set(4,480) 
feed.set(10,100) 

lefteye = [362, 385, 387, 263]  # Outer left eye
righteye = [33, 160, 158, 133]  # Outer right eye

def eye_aspect_ratio(landmarks, eye):
    left = landmarks[eye[0]]
    right = landmarks[eye[3]]
    top = landmarks[eye[1]]
    bottom = landmarks[eye[2]]

    hor = math.hypot(left.x - right.x, left.y - right.y)
    ver = math.hypot(top.x - bottom.x, top.y - bottom.y)

    return ver / hor if hor != 0 else 0

#DEFAULT FORMALITY!!!!
mphands=mp.solutions.hands
hands=mphands.Hands()             
mpdraw=mp.solutions.drawing_utils 

mp_face = mp.solutions.face_mesh
facemesh = mp_face.FaceMesh(refine_landmarks=True)


while True:
    success,img=feed.read()
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
    results=hands.process(imgrgb)       
    '''print(results.multi_hand_landmarks)'''

    if results.multi_hand_landmarks:
        for handlandmarks in results.multi_hand_landmarks:
            for id,landmarks in enumerate(handlandmarks.landmark): 
                '''print(id,landmarks)'''  
                height,width,channel=img.shape  
                pixelx,pixely=int(landmarks.x*width), int(landmarks.y*height)

                if id in [8] :     
                    cv2.circle(img, (pixelx,pixely), 15, (255, 0, 255), cv2.FILLED) 


            mpdraw.draw_landmarks(img,handlandmarks,mphands.HAND_CONNECTIONS)

    faceresult = facemesh.process(imgrgb)
    blink = False 
    if faceresult.multi_face_landmarks:
        for face_landmarks in faceresult.multi_face_landmarks:
            l_ratio = eye_aspect_ratio(face_landmarks.landmark, lefteye)
            r_ratio = eye_aspect_ratio(face_landmarks.landmark, righteye)
            avg = (l_ratio + r_ratio) / 2

            if avg < 0.20:
                blink = True
                cv2.putText(img, "BLINK!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    img=cv2.flip(img,1)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break