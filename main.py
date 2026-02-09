import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import time
import numpy as np
from typing import Optional, List, Union

class CameraStream:
    def __init__(self, source: Union[int, str]) -> cv2.typing.MatLike | None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera source: {source}")
        
    def read(self) -> Optional[cv2.typing.MatLike]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

class FaceDetector:
    def __init__(self, model_selection:int=1, min_confidence:float=0.6):
        self.mp_face_detection = mp.tasks.vision.FaceDetector

        # Model selection 0 = Short Range (WebCam), 1 = Long Range (IP Cam)
        self.detector = mp.tasks.vision.FaceDetectorOptions(
            model_selection = model_selection, min_detection_confidence = min_confidence
        )
    def detect(self, frame):
        if frame is None:
            return []
        
        # 1. Converts BGR (OpenCV) to RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.detector.process(frame_rgb)

        faces = []
        if results.detections:
            for det in results.detections:
                faces.append(det.location_data.relative_bounding_box)
        return faces
    
    def close(self):
        self.detector.close()



def main():
    cam = CameraStream(source=0)
    pose_scan = PoseDetector(model_path='pose_landmarker_lite.task')
    print("Running... Press 'q' to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is not None: 
                break

            if pose_scan.latest_result and pose_scan.latest_result.pose_landmarks:
                nose_x = pose_scan.latest_result.pose_landmarks[0][0].x
                print(f"Nose X: {nose_x:.4f}")
            
            cv2.imshow('Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cam.release()
        pose_scan.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
