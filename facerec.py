import cv2
import mediapipe as mp
import math
import time
import numpy as np
from thakkalimeter import estimate_tomatoes

feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
feed.set(3, 640)
feed.set(4, 480)

lefteye = [362, 385, 387, 263]
righteye = [33, 160, 158, 133]

mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_mesh
facemesh = mp_face.FaceMesh(refine_landmarks=True)

def eye_aspect_ratio(landmarks, eye):
    left = landmarks[eye[0]]
    right = landmarks[eye[3]]
    top = landmarks[eye[1]]
    bottom = landmarks[eye[2]]
    hor = math.hypot(left.x - right.x, left.y - right.y)
    ver = math.hypot(top.x - bottom.x, top.y - bottom.y)
    return ver / hor if hor != 0 else 0

# Load tomato image (make sure it's in same folder)
tomato_img = cv2.imread("tomato.png", cv2.IMREAD_UNCHANGED)
tomato_img = cv2.resize(tomato_img, (40, 40))  # size of each tomato

blink_display_time = 0
blink_cooldown = 0
photo_taken = False
overlay_tomatoes = False
tomato_positions = []

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
                    photo_taken = False

    show_blink = time.time() - blink_display_time < 1

    if index_detected:
        cv2.putText(img, "Index Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if index_detected and show_blink:
        cv2.putText(img, "BLINKED!", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if not photo_taken:
            filename = "captured_face.jpg"
            cv2.imwrite(filename, img)
            count, face_box = estimate_tomatoes(filename)
            print(f"🍅 Tomatoes on face: {count}")
            overlay_tomatoes = True
            tomato_positions = []

            if face_box:
                x, y, w_, h_ = face_box
                for i in range(count):
                    xi = x + (i % 5) * 45
                    yi = y + (i // 5) * 45
                    tomato_positions.append((xi, yi))

            photo_taken = True

    # Overlay tomatoes
    if overlay_tomatoes:
        for (xi, yi) in tomato_positions:
            if xi + 40 > img.shape[1] or yi + 40 > img.shape[0]:
                continue
            roi = img[yi:yi+40, xi:xi+40]
            alpha_s = tomato_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                roi[:, :, c] = (alpha_s * tomato_img[:, :, c] +
                                alpha_l * roi[:, :, c])
            img[yi:yi+40, xi:xi+40] = roi

    cv2.imshow("TomatoFace™", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
