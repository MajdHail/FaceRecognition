import cv2
import time
import mediapipe as mp
import numpy as np

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam setup
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    direction = "NO FACE"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        nose = landmarks[1]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        nose_x = nose.x * w
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w

        face_center = (left_x + right_x) / 2
        offset = nose_x - face_center
        threshold = w * 0.02

        if offset > threshold:
            direction = "LEFT"
        elif offset < -threshold:
            direction = "RIGHT"
        else:
            direction = "FRONT"

        cv2.circle(frame, (int(nose_x), int(nose.y * h)), 5, (0, 0, 255), -1)

    cv2.putText(
        frame,
        direction,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Face Direction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

