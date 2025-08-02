import cv2
import mediapipe as mp
import math
import time

feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
feed.set(3, 640)
feed.set(4, 480)
feed.set(10, 200)

lefteye = [362, 385, 387, 263]
righteye = [33, 160, 158, 133]

def eye_aspect_ratio(landmarks, eye):
    left = landmarks[eye[0]]
    right = landmarks[eye[3]]
    top = landmarks[eye[1]]
    bottom = landmarks[eye[2]]
    hor = math.hypot(left.x - right.x, left.y - right.y)
    ver = math.hypot(top.x - bottom.x, top.y - bottom.y)
    return ver / hor if hor != 0 else 0

mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_mesh
facemesh = mp_face.FaceMesh(refine_landmarks=True)

blink_display_time = 0
blink_cooldown = 0
index_detected = False
photo_taken = False  # flag to avoid saving multiple photos on same blink

while True:
    success, img = feed.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    index_detected = False

    hand_results = hands.process(imgrgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    index_detected = True
            mpdraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)

    face_results = facemesh.process(imgrgb)
    blinked = False

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:

            for idx in lefteye + righteye:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            l_ratio = eye_aspect_ratio(face_landmarks.landmark, lefteye)
            r_ratio = eye_aspect_ratio(face_landmarks.landmark, righteye)
            avg_ratio = (l_ratio + r_ratio) / 2

            if avg_ratio < 0.38:
                if time.time() - blink_cooldown > 1:
                    blink_display_time = time.time()
                    blink_cooldown = time.time()
                    blinked = True
                    photo_taken = False  # allow photo again after new blink

    show_blink = time.time() - blink_display_time < 1

    if index_detected:
        cv2.putText(img, "Index Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if index_detected and show_blink:
        cv2.putText(img, "BLINKED!", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if not photo_taken:
            timestamp = int(time.time())
            filename = f"snap_{timestamp}.jpg"
            cv2.imwrite(filename, img)
            print(f"Photo saved: {filename}")
            photo_taken = True

    cv2.imshow("Blink + Index Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
