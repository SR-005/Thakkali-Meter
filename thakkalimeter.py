import cv2
import mediapipe as mp
import math

def estimate_tomatoes(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image.")
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        face_width = x_max - x_min
        face_height = y_max - y_min

        # Dynamically choose tomato radius (fit approx 6–10 in a row)
        tomatoes_per_row = 6
        tomato_radius = 20
        tomato_diameter = tomato_radius * 2

        cols = face_width // tomato_diameter
        rows = face_height // tomato_diameter
        total_tomatoes = cols * rows

        # Draw tomato circles over the face
        for i in range(rows):
            for j in range(cols):
                cx = x_min + j * tomato_diameter + tomato_radius
                cy = y_min + i * tomato_diameter + tomato_radius
                cv2.circle(image, (cx, cy), tomato_radius, (0, 0, 255), -1)

        # Draw bounding box and label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'{total_tomatoes} tomatoes fit!', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Thakkalimeter™", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected.")

# Test
if __name__ == "__main__":
    estimate_tomatoes("face.jpg")  # replace with captured image path
