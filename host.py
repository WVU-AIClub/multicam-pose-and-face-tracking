# This program is meant to be run on the host device (main laptop) that will receive video streams from other devices

import math
import numpy as np
import cv2
import imagezmq
import time
import mediapipe as mp
from face_tracker import FaceTracker
from pose_tracker import PoseTracker
from multi_tracking import draw_faces, draw_poses

def create_image_grid(images, tile_size=(320, 240)):
    """
    Takes a list of images, resizes them to tile_size, 
    and stacks them into a square-ish grid.
    """
    if not images:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)

    # 1. Resize all images to a common size so they stack correctly
    resized_imgs = [cv2.resize(img, tile_size) for img in images]
    
    # 2. Calculate Grid Dimensions (Columns & Rows)
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    # 3. Fill empty slots with black images if grid isn't full
    empty_slots = (cols * rows) - n
    if empty_slots > 0:
        black_tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        for _ in range(empty_slots):
            resized_imgs.append(black_tile)
            
    # 4. Build the Grid using NumPy
    # Split the list into rows
    grid_rows = []
    for r in range(rows):
        # Get the images for this row
        row_imgs = resized_imgs[r*cols : (r+1)*cols]
        # Stack them horizontally (hstack)
        row_concat = np.hstack(row_imgs)
        grid_rows.append(row_concat)
        
    # Stack the rows vertically (vstack)
    final_grid = np.vstack(grid_rows)
    return final_grid

# Initialize the Hub (host)
image_hub = imagezmq.ImageHub()

# Dictionary to store both trackers for each camera
# Structure: { 'CamName': {'face': FaceTrackerObj, 'pose': PoseTrackerObj} }
device_trackers = {}

# Store the most recent frame from each camera
latest_frames = {}

print("Dashboard Hub Started. Waiting for cameras to connect...")

try:
    while True:
        # 3. Receive frame (Blocks until data arrives)
        cam_name, frame = image_hub.recv_image()
        
        # 4. Initialization: Create BOTH trackers if this is a new camera
        if cam_name not in device_trackers:
            print(f"[INFO] New camera detected: {cam_name}")
            device_trackers[cam_name] = {
                'face': FaceTracker(model_path='models/detector.tflite'),
                'pose': PoseTracker(model_path='models/pose_landmarker_lite.task', num_poses=10)
            }
            latest_frames[cam_name] = frame
            
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
        face_result = toolkit['face'].get_latest_result()
        pose_result = toolkit['pose'].get_latest_result()


        if pose_result and pose_result.pose_landmarks:
            draw_poses(pose_result, frame)
        
        if face_result and face_result.detections:
            draw_faces(face_result, frame)
        
        latest_frames[cam_name] = frame

        current_images = list(latest_frames.values())
        dashboard = create_image_grid(current_images, tile_size=(400,300))

        # 8. Show the window
        cv2.imshow("Multi-Camera Dashboard", dashboard)
        
        # 9. Send Reply (Required)
        image_hub.send_reply(b'OK')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Quitting Program")


finally:
    # Clean up all trackers for all devices
    for toolkit in device_trackers.values():
        toolkit['face'].close()
        toolkit['pose'].close()
    cv2.destroyAllWindows()