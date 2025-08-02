import cv2
import mediapipe as mp
import math
import time
import numpy as np
from thakkalimeter import estimate_tomatoes

lefteye = [362, 385, 387, 263]
righteye = [33, 160, 158, 133]

mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_mesh
facemesh = mp_face.FaceMesh(refine_landmarks=True)

# Load tomato image (make sure it's in same folder)
tomato_img = cv2.imread("tomato.png", cv2.IMREAD_UNCHANGED)
tomato_img = cv2.resize(tomato_img, (40, 40))  # size of each tomato

def eye_aspect_ratio(landmarks, eye):
    left = landmarks[eye[0]]
    right = landmarks[eye[3]]
    top = landmarks[eye[1]]
    bottom = landmarks[eye[2]]
    hor = math.hypot(left.x - right.x, left.y - right.y)
    ver = math.hypot(top.x - bottom.x, top.y - bottom.y)
    return ver / hor if hor != 0 else 0

def generate_frames():
    feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    feed.set(3, 640)
    feed.set(4, 480)

    blink_display_time = 0
    blink_cooldown = 0
    photo_taken = False

    while True:
        success, img = feed.read()
        if not success:
            break

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

        if index_detected and show_blink and not photo_taken:
            cv2.putText(img, "BLINKED!", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            filename = "captured_face.jpg"
            cv2.imwrite(filename, img)
            feed.release()
            cv2.destroyAllWindows()

            count, face_box = estimate_tomatoes(filename)
            print(f"🍅 Tomatoes on face: {count}")
            return count  # 🟢 Return count immediately

        cv2.imshow("TomatoFace™ - Blink & Raise Finger", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    feed.release()
    cv2.destroyAllWindows()
    return 0  # Default if nothing captured

if __name__ == "__main__":
    generate_frames()
