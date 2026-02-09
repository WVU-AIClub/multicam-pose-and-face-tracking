import mediapipe as mp
import time

class PoseDetector:
    # Model Types: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models
    # Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream

    def __init__(self, model_path:str, num_poses:int=1, min_confidence:float=0.5):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=model_path),
            running_mode = VisionRunningMode.LIVE_STREAM,
            num_poses = num_poses,
            min_tracking_confidence = min_confidence,
            result_callback = self._result_callback
        )
        
        self.detector = PoseLandmarker.create_from_options(self.options)
        self.latest_result = None

    def _result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    def detect(self, frame):
        if frame is None: return
        
        # Conver to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.detector.detect_async(mp_image, int(time.time()*1000))

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