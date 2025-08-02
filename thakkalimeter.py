# tomato_face.py
import cv2
import mediapipe as mp
import math

def estimate_tomatoes(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        face_area = (x_max - x_min) * (y_max - y_min)

        tomato_radius = 20
        tomato_area = math.pi * (tomato_radius ** 2)
        tomato_count = int(face_area / tomato_area)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'{tomato_count} tomatoes fit!', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Tomato Estimation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

