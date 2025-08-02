import cv2
import mediapipe as mp
import math
import time

# Setup camera
feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
feed.set(3, 640)
feed.set(4, 480)
feed.set(10, 170)

# Eye landmark indices (outer corners and lids)
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

# MediaPipe setup
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_mesh
facemesh = mp_face.FaceMesh(refine_landmarks=True)

# State tracking
blink_display_time = 0
blink_cooldown = 0
index_detected = False

while True:
    success, img = feed.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    index_detected = False

    # Hand tracking
    hand_results = hands.process(imgrgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    index_detected = True
            mpdraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)

    # Face + blink tracking
    face_results = facemesh.process(imgrgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw eye dots only
            for idx in lefteye + righteye:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            # EAR detection
            l_ratio = eye_aspect_ratio(face_landmarks.landmark, lefteye)
            r_ratio = eye_aspect_ratio(face_landmarks.landmark, righteye)
            avg_ratio = (l_ratio + r_ratio) / 2

            if avg_ratio < 0.38:
                if time.time() - blink_cooldown > 1:
                    blink_display_time = time.time()
                    blink_cooldown = time.time()

    # Determine display
    show_blink = time.time() - blink_display_time < 1

    if index_detected:
        cv2.putText(img, "Index finger detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if index_detected and show_blink:
        cv2.putText(img, "BLINKED!", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # Show result
    cv2.imshow("Blink + Index Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
