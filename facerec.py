import cv2
import mediapipe as mp
import math
import time

# Setup camera
feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
feed.set(3, 640)
feed.set(4, 480)
feed.set(10, 170)

# Eye landmark indices
lefteye = [362, 385, 387, 263]
righteye = [33, 160, 158, 133]

# EAR calculation
def eye_aspect_ratio(landmarks, eye):
    left = landmarks[eye[0]]
    right = landmarks[eye[3]]
    top = landmarks[eye[1]]
    bottom = landmarks[eye[2]]

    hor = math.hypot(left.x - right.x, left.y - right.y)
    ver = math.hypot(top.x - bottom.x, top.y - bottom.y)

    return ver / hor if hor != 0 else 0

# Init MediaPipe
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_mesh
facemesh = mp_face.FaceMesh(refine_landmarks=True)

# Blink logic state
blink_display_time = 0
blink_cooldown = 0
index_detected = False
blinked = False

while True:
    success, img = feed.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reset flags each frame
    index_detected = False
    blinked = False

    # Process hands
    hand_results = hands.process(imgrgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:  # Index fingertip
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    index_detected = True
            mpdraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)

    # Process face and detect blink
    face_results = facemesh.process(imgrgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            h, w, _ = img.shape
            mpdraw.draw_landmarks(img, face_landmarks, mp_face.FACEMESH_TESSELATION)

            # Draw green circles on eye landmarks
            for idx in lefteye + righteye:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

            # Calculate EAR
            l_ratio = eye_aspect_ratio(face_landmarks.landmark, lefteye)
            r_ratio = eye_aspect_ratio(face_landmarks.landmark, righteye)
            avg_ratio = (l_ratio + r_ratio) / 2

            # Print with emoji
            if avg_ratio < 0.34:
                print(f"🔴 Blink detected! EAR: {avg_ratio:.3f}")
            else:
                print(f"🟢 Eyes open. EAR: {avg_ratio:.3f}")

            # Show EAR for reference
            cv2.putText(img, f"EAR: {avg_ratio:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Blink detection logic
            if avg_ratio < 0.36:
                if time.time() - blink_cooldown > 1:
                    blink_display_time = time.time()
                    blink_cooldown = time.time()

    # Determine what to show
    show_blink = time.time() - blink_display_time < 1

    # Status messages
    if index_detected:
        cv2.putText(img, "Index finger detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if show_blink:
        cv2.putText(img, "Blink detected", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if index_detected and show_blink:
        cv2.putText(img, "BLINKED!", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # Show video
    cv2.imshow("TomatoFace – Blink & Index Finger Status", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
