import mediapipe as mp
import time

class PoseTracker:
    # Model Types: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models
    # Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream

    def __init__(self, model_path:str, num_poses:int=1, min_confidence:float=0.5):
        self.latest_result = None

        def _result_callback(result, output_image: mp.Image, timestamp_ms: int):
            self.latest_result = result

        # Configure options
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=model_path),
            running_mode = VisionRunningMode.LIVE_STREAM,
            num_poses = num_poses,
            min_tracking_confidence = min_confidence,
            result_callback = _result_callback
        )
        
        # Initialize the detector
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, mp_image, timestamp_ms):
        """
        Sends the frame to MediaPipe for processing.
        This is non-blocking (async).
        """

        return self.detector.detect_async(mp_image, timestamp_ms)
    
    def get_latest_result(self):
        """
        Returns the most recent result received by the callback.
        """
        return self.latest_result

    def close(self):
        self.detector.close()