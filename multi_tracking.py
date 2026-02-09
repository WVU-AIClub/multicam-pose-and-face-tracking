import cv2
import time
import mediapipe as mp
from pose_tracker import PoseTracker
from face_tracker import FaceTracker

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (11, 23), (12, 24), (23, 24),                      # Torso
    (23, 25), (24, 26), (25, 27), (26, 28),            # Legs
    (27, 29), (28, 30), (29, 31), (30, 32)             # Feet
]

def draw_poses(pose_result):
    for landmarks in pose_result.pose_landmarks: # NOTE: Ignore this fake error, it thinks face_result is always None
        for landmark in landmarks:
            h, w, _ = frame.shape

            # A. Draw Lines (Skeleton)
            for start_idx, end_idx in POSE_CONNECTIONS:
                start_pt = landmarks[start_idx]
                end_pt = landmarks[end_idx]
                
                x1, y1 = int(start_pt.x * w), int(start_pt.y * h)
                x2, y2 = int(end_pt.x * w), int(end_pt.y * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # B. Draw Points (Joints)
            for landmark in landmarks:
                cx, cy = int(landmark.x * w) , int(landmark.y *h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

def draw_faces(face_result):
    if face_result and face_result.detections:
        for detection in face_result.detections: # NOTE: Ignore this fake error, it thinks face_result is always None
            bbox = detection.bounding_box
            cv2.rectangle(frame, 
                            (bbox.origin_x, bbox.origin_y), 
                            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), 
                            (0, 255, 0), 2)
            cv2.putText(frame, "Face", (bbox.origin_x, bbox.origin_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 1. Instantiate your tracker classes
pose_tracker = PoseTracker(model_path='models/pose_landmarker_lite.task', num_poses= 10)
face_tracker = FaceTracker(model_path='models/detector.tflite') # Your future Pose class

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Shared Pre-Processing
        # Convert once for all trackers to save processing time
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        # Dispatch to Trackers
        # These run asynchronously!
        pose_tracker.process_frame(mp_image, timestamp_ms)
        face_tracker.process_frame(mp_image, timestamp_ms) 

        # Retrieve & Visualize Results
        # Check Face Results
        face_result = face_tracker.get_latest_result()
        pose_result = pose_tracker.get_latest_result()


        if pose_result and pose_result.pose_landmarks:
            draw_poses(pose_result)
        
        if face_result and face_result.detections:
            draw_faces(face_result)


        cv2.imshow('Multi-Tracker MERGE', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 5. Clean up
    pose_tracker.close()
    face_tracker.close()
    cap.release()
    cv2.destroyAllWindows()