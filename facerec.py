import cv2
import mediapipe as mp

feed = cv2.VideoCapture(0, cv2.CAP_DSHOW) #if you have only one webcam- 0 reads feed fom that one
feed.set(3,640) #setting up its width - id 3 refers to width
feed.set(4,480) #setting up its height - id 4 refers to height
feed.set(10,100) #id 10 is for BRIGHTNESS 

#DEFAULT FORMALITY!!!!
mphands=mp.solutions.hands
hands=mphands.Hands()               #it can contain multiple hands
mpdraw=mp.solutions.drawing_utils  #built in funtion to point hand landmarks for each hand(in this case)

while True:
    success,img=feed.read()
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #Converting BGR color coded img to RGB as it is the one supported!
    results=hands.process(imgrgb)       #"process" is an inbuilt function that gives the necessart details automatically
    '''print(results.multi_hand_landmarks)'''

    if results.multi_hand_landmarks:
        for handlandmarks in results.multi_hand_landmarks:
            for id,landmarks in enumerate(handlandmarks.landmark): #getting id and landmarks of the hands in the feed [DATA]
                '''print(id,landmarks)'''  #returns the position of landmarks in decimal values, we have to convert to pixel values
                height,width,channel=img.shape   #collecting height,width and channel inorder to calculate position in pixels i,e x*width and y*height
                pixelx,pixely=int(landmarks.x*width), int(landmarks.y*height)
                print(id,pixelx,pixely)

                if id in [8] :     #drawing a purple circle with landmark 0
                    cv2.circle(img, (pixelx,pixely), 15, (255, 0, 255), cv2.FILLED) #5 is the radius and (255,0,255) is code for color purple


            mpdraw.draw_landmarks(img,handlandmarks,mphands.HAND_CONNECTIONS) #in the "img" it will set landmarks for each hand in the feed and set connections

    img=cv2.flip(img,1)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break