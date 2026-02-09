import cv2
import time
import mediapipe as mp
import numpy as np
import pickle
import os

mp_face_mesh = mp.solutions.face_mesh


class FaceRecognitionSystem:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Storage for known faces
        self.known_faces = {}  # {name: [embeddings]}
        self.database_file = "face_database.pkl"
        self.load_database()

        # Registration state
        self.registration_mode = False
        self.registration_name = ""
        self.registration_step = 0  # 0: left, 1: front, 2: right
        self.registration_data = []

    def load_database(self):
        """Load the face database from file"""
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"Loaded {len(self.known_faces)} people from database")

    def save_database(self):
        """Save the face database to file"""
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print("Database saved")

    def get_face_embedding(self, landmarks, w, h):
        """Extract a simple embedding from face landmarks"""
        # Use key facial landmarks to create an embedding
        key_points = [1, 33, 61, 199, 263, 291, 362, 454, 234]  # nose, eyes, mouth corners, cheeks
        embedding = []

        for idx in key_points:
            landmark = landmarks[idx]
            embedding.extend([landmark.x, landmark.y, landmark.z])

        return np.array(embedding)

    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings using Euclidean distance"""
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance

    def recognize_face(self, embedding, threshold=0.15):
        """Try to recognize a face from the database"""
        best_match = None
        best_distance = float('inf')

        for name, embeddings in self.known_faces.items():
            for known_embedding in embeddings:
                distance = self.compare_faces(embedding, known_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name

        if best_distance < threshold:
            return best_match, best_distance
        return None, best_distance

    def add_person(self, name, embedding):
        """Add a new person to the database"""
        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].append(embedding)

    def start_registration(self, name):
        """Start the registration process for a new person"""
        self.registration_mode = True
        self.registration_name = name
        self.registration_step = 0
        self.registration_data = []
        print(f"Starting registration for: {name}")

    def process_registration_step(self, embedding, direction):
        """Process one step of the registration"""
        steps = ["LEFT", "FRONT", "RIGHT"]

        if direction == steps[self.registration_step]:
            self.registration_data.append(embedding)
            print(f"Captured {steps[self.registration_step]} view")
            self.registration_step += 1

            if self.registration_step >= 3:
                # Registration complete
                for emb in self.registration_data:
                    self.add_person(self.registration_name, emb)
                self.save_database()
                print(f"Registration complete for {self.registration_name}!")
                self.registration_mode = False
                return True

        return False

    def get_registration_instruction(self):
        """Get the current instruction for registration"""
        steps = ["Turn LEFT", "Look FRONT", "Turn RIGHT"]
        if self.registration_step < 3:
            return steps[self.registration_step]
        return "Complete!"


def main():
    system = FaceRecognitionSystem()

    webcam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    time.sleep(1)

    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam")

    print("\n=== Face Recognition System ===")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Register new person")
    print("  'd' - Delete all registered faces")
    print("  'l' - List registered people")
    print("================================\n")

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = system.face_mesh.process(rgb)

        direction = "NO FACE"
        recognized_name = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Detect orientation
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

            # Get face embedding
            embedding = system.get_face_embedding(landmarks, w, h)

            # Registration mode
            if system.registration_mode:
                if system.process_registration_step(embedding, direction):
                    # Registration complete
                    pass
            else:
                # Recognition mode
                recognized_name, distance = system.recognize_face(embedding)

            # Draw nose marker
            cv2.circle(frame, (int(nose_x), int(nose.y * h)), 5, (0, 0, 255), -1)

        # Display information
        y_offset = 40

        # Display mode
        if system.registration_mode:
            mode_text = f"REGISTRATION: {system.registration_name}"
            instruction = system.get_registration_instruction()
            cv2.putText(frame, mode_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 165, 255), 2)
            y_offset += 35
            cv2.putText(frame, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)
            y_offset += 40
        else:
            # Display recognized name in top right
            if recognized_name:
                name_text = recognized_name
                text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = w - text_size[0] - 20
                cv2.putText(frame, name_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)

        # Display direction
        cv2.putText(frame, direction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3)

        cv2.imshow("Face Recognition System", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r') and not system.registration_mode:
            # Start registration
            name = input("\nEnter name to register: ").strip()
            if name:
                system.start_registration(name)
                print("Look at the camera and follow the instructions...")
        elif key == ord('d'):
            # Delete database
            confirm = input("\nDelete all registered faces? (yes/no): ").strip().lower()
            if confirm == 'yes':
                system.known_faces = {}
                system.save_database()
                print("Database cleared!")
        elif key == ord('l'):
            # List registered people
            print(f"\nRegistered people ({len(system.known_faces)}):")
            for name, embeddings in system.known_faces.items():
                print(f"  - {name} ({len(embeddings)} views)")

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()