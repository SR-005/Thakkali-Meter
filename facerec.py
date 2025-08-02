import cv2
import mediapipe as mp

feed = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
feed.set(3,640) 
feed.set(4,480) 
feed.set(10,100) 

lefteye = [362, 385, 387, 263]  # Outer left eye
righteye = [33, 160, 158, 133]  # Outer right eye

#DEFAULT FORMALITY!!!!
mphands=mp.solutions.hands
hands=mphands.Hands()             
mpdraw=mp.solutions.drawing_utils  

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

    img=cv2.flip(img,1)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break