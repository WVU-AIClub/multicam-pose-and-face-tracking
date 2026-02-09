import cv2
import time
import mediapipe as mp
from face_tracker import FaceTracker


# 1. Instantiate your tracker classes
face_tracker = FaceTracker(model_path='models/detector.tflite')
# pose_tracker = PoseTracker(...) # Your future Pose class

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 2. Shared Pre-processing
        # Convert once for all trackers to save processing time
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        # 3. Dispatch to Trackers
        # These run asynchronously!
        face_tracker.process_frame(mp_image, timestamp_ms)
        # pose_tracker.process_frame(mp_image, timestamp_ms) 

        # 4. Retrieve & Visualize Results
        # Check Face Results
        face_result = face_tracker.get_latest_result()
        if face_result and face_result.detections:
            for detection in face_result.detections: # NOTE: Ignore this fake error, it thinks face_result is always None
                bbox = detection.bounding_box
                cv2.rectangle(frame, 
                             (bbox.origin_x, bbox.origin_y), 
                             (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), 
                             (0, 255, 0), 2)
                cv2.putText(frame, "Face", (bbox.origin_x, bbox.origin_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check Pose Results (Example)
        # pose_result = pose_tracker.get_latest_result()
        # if pose_result:
        #    ... draw pose skeleton ...

        cv2.imshow('Multi-Tracker Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 5. Clean up
    face_tracker.close()
    # pose_tracker.close()
    cap.release()
    cv2.destroyAllWindows()