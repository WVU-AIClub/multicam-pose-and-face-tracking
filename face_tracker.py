# Download the FaceDetector model (already in the models/ directory)
# wget -O detector.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

import mediapipe as mp
import time

class FaceTracker:
    def __init__(self, model_path: str = 'models/detector.tflite', min_confidence: float = 0.6):
        self.latest_result = None
        
        # Define the callback inside the class
        def _result_callback(result, output_image, timestamp_ms):
            self.latest_result = result

        # Configure options
        BaseOptions = mp.tasks.BaseOptions
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_detection_confidence=min_confidence,
            result_callback=_result_callback
        )
        
        # Initialize the detector
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def process_frame(self, mp_image, timestamp_ms):
        """
        Sends the frame to MediaPipe for processing. 
        This is non-blocking (async).
        """
        self.detector.detect_async(mp_image, timestamp_ms)

    def get_latest_result(self):
        """
        Returns the most recent result received by the callback.
        """
        return self.latest_result

    def close(self):
        self.detector.close()