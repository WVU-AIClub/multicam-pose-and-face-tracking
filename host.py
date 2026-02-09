# This program is meant to be run on the host device (main laptop) that will receive video streams from other devices

import cv2
import imagezmq
import time
import mediapipe as mp
from face_tracker import FaceTracker
from pose_tracker import PoseDetector

# Initialize the Hub (host)
image_hub = imagezmq.ImageHub()

# Dictionary to store both trackers for each camera
# Structure: { 'CamName': {'face': FaceTrackerObj, 'pose': PoseTrackerObj} }
device_trackers = {} 

print("Waiting for cameras to connect...")

try:
    while True:
        # 3. Receive frame (Blocks until data arrives)
        cam_name, frame = image_hub.recv_image()
        
        # 4. Initialization: Create BOTH trackers if this is a new camera
        if cam_name not in device_trackers:
            print(f"[INFO] New camera detected: {cam_name}")
            device_trackers[cam_name] = {
                'face': FaceTracker(model_path='models/detector.tflite'),
                'pose': PoseDetector(model_path='models/pose_landmarker_lite.task')
            }
            
        # 5. Get the specific toolkit for this camera
        toolkit = device_trackers[cam_name]
        
        # Prepare image ONCE for both models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        # 6. Dispatch to BOTH models
        # Note: This is still efficient because detect_async returns immediately
        toolkit['face'].process_frame(mp_image, timestamp_ms)
        toolkit['pose'].process_frame(mp_image, timestamp_ms)
        
        # 7. Visualization Loop
        
        # --- Draw FACE Results ---
        face_result = toolkit['face'].get_latest_result()
        if face_result and face_result.detections:
            for detection in face_result.detections:
                bbox = detection.bounding_box
                cv2.rectangle(frame, 
                              (bbox.origin_x, bbox.origin_y), 
                              (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), 
                              (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {cam_name}", (bbox.origin_x, bbox.origin_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Draw POSE Results ---
        pose_result = toolkit['pose'].get_latest_result()
        if pose_result and pose_result.pose_landmarks:
            for landmarks in pose_result.pose_landmarks:
                for landmark in landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # 8. Show the window
        # Note: 'cv2.imshow' creates a window named after the camera
        # If you have 3 cameras, you will get 3 pop-up windows!
        cv2.imshow(f"Stream: {cam_name}", frame)
        
        # 9. Send Reply (Required)
        image_hub.send_reply(b'OK')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up all trackers for all devices
    for toolkit in device_trackers.values():
        toolkit['face'].close()
        toolkit['pose'].close()
    cv2.destroyAllWindows()