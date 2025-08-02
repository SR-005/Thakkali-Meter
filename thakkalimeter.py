import cv2
import mediapipe as mp
import math
import numpy as np

def overlay_image(bg, overlay, x, y):
    h, w, _ = overlay.shape
    bg_h, bg_w = bg.shape[:2]
    if x + w > bg_w or y + h > bg_h:
        return bg
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                               alpha_bg * bg[y:y+h, x:x+w, c])
    return bg

def estimate_tomatoes(image_path, tomato_path="tomato.png", tomato_radius = 20):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
        return 0, None

    h, w, _ = image.shape
    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

    # Create face mask using convex hull
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    x_min, y_min, face_width, face_height = cv2.boundingRect(hull)
    # Expand bounding box vertically
    padding_y = int(face_height * 0.2)  # Increase this value if needed
    y_min = max(0, y_min - padding_y)
    face_height = min(h - y_min, face_height + padding_y * 2)
    tomato_radius = int(min(face_width, face_height) / 4)

    # Load and resize tomato image
    tomato_img = cv2.imread(tomato_path, cv2.IMREAD_UNCHANGED)
    tomato_img = cv2.resize(tomato_img, (tomato_radius*2, tomato_radius*2))

    positions = []
    step = tomato_radius * 2
    for y in range(y_min, y_min + face_height, step):
        for x in range(x_min, x_min + face_width, step):
            cx = x + tomato_radius
            cy = y + tomato_radius
            if cx >= w or cy >= h:
                continue
            if mask[cy, cx] == 255:
                image = overlay_image(image, tomato_img, x, y)
                positions.append((x, y))

    cv2.putText(image, f'{len(positions)} tomatoes fit!', (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Thakkali Meter", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(positions), (x_min, y_min, face_width, face_height)
