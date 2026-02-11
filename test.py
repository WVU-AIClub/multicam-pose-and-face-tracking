import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from face_tracker import FaceTracker
from multi_tracking import draw_faces, draw_poses
import os, time
"""
Biometric Signature
    1. Face: A small crop of the face is passed to a library like 
    DeepFace to generate a unqiue numerical vector.
    2. Pose: We calculate Scale-Invariant Ratios (shoulder-width to height). These stay
    consistent even if the person moves closer or farther.
    """

# Initialize MediaPipe Holstic
# mp_holisitic = mp.tasks.
# holistic = mp_holistic.Holistic(min_detection_confidience=0.5, min_tracking_confidence=0.5)
def main():
    # Setup paths
    DB_PATH = "biometric_db"
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)

    face_tracker = FaceTracker(model_path='models/detector.tflite')

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Shared Pre-Processing
            # Convert once for all trackers to save processing time
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            # Dispatch to Trackers
            # These run asynchronously!
            face_tracker.process_frame(mp_image, timestamp_ms) 

            # Retrieve & Visualize Results
            # Check Face Results
            face_result = face_tracker.get_latest_result()

            # Layer 1: Try Face Recognition
            label = "Searching..."
            color = (255, 0, 0)
            if face_result and face_result.detections is not []:
                face_objs = DeepFace.represent(img_path=frame, enforce_detection=False)
                print(face_objs)


                if len(face_objs) > 0:
                    label = face_objs[0]['identity'][0].split('/')[-1].split('.')[0]
                    color = (0, 255, 0)
                else:
                    label = "New Face Detected"
                    DeepFace.register(img=frame)

            cv2.putText(frame, f"Identity: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        
            if face_result and face_result.detections:
                draw_faces(face_result, frame)


            cv2.imshow('Multi-Tracker MERGE', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 5. Clean up
        face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()

def get_body_ratios(landmarks):
    """ Calculate bone length ratios taht are unique to a person's frame."""
    # Landmark IDs: 11, 12 (Shoulders), 23, 24 (Hips)
    sh_width = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - 
                                np.array([landmarks[12].x, landmarks[12].y]))
    
    torso_height = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - 
                                  np.array([landmarks[23].x, landmarks[23].y]))
    
    return sh_width / torso_height  # Shoulder-to-Torso Ratio

if "__main__" == __name__:
    main()    