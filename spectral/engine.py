import time
import cv2
import numpy as np
from typing import Optional, Callable, Dict, List

from spectral.models import (
    FaceData,
    TrackingFrame,
    TrackingConfig,
    GestureType,
    GestureEvent,
)
from spectral.core.detector import FaceDetector
from spectral.core.pose import PoseEstimator
from spectral.core.expression import ExpressionAnalyzer
from spectral.core.gestures import GestureDetector
from spectral.tracking.eye_tracker import EyeTracker
from spectral.tracking.eyebrow_tracker import EyebrowTracker
from spectral.tracking.nose_tracker import NoseTracker
from spectral.tracking.mouth_tracker import MouthTracker
from spectral.tracking.jaw_tracker import JawTracker
from spectral.utils.visualization import Visualizer, DashboardVisualizer


class SpectralEngine:
    def __init__(self, config: Optional[TrackingConfig] = None):
        self._config = config or TrackingConfig()
        self._detector = FaceDetector(self._config)
        self._eye_tracker = EyeTracker(self._config)
        self._eyebrow_tracker = EyebrowTracker(self._config)
        self._nose_tracker = NoseTracker(self._config)
        self._mouth_tracker = MouthTracker(self._config)
        self._jaw_tracker = JawTracker(self._config)
        self._pose_estimator = PoseEstimator()
        self._expression_analyzer = ExpressionAnalyzer(self._config)
        self._gesture_detector = GestureDetector(self._config)
        self._visualizer = Visualizer()
        self._dashboard = DashboardVisualizer()
        self._frame_count = 0
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._capture = None

    @property
    def config(self) -> TrackingConfig:
        return self._config

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running

    def process_frame(self, frame: np.ndarray) -> TrackingFrame:
        start_time = time.perf_counter()
        self._frame_count += 1

        results, width, height = self._detector.detect(frame)
        face_data_list = self._detector.extract_landmarks(results, width, height)

        faces = []
        for face_id, (landmarks_2d, landmarks_3d, bbox, confidence) in enumerate(
            face_data_list
        ):
            left_eye = None
            right_eye = None
            if self._config.enable_eye_tracking:
                left_eye = self._eye_tracker.track_left_eye(landmarks_2d)
                right_eye = self._eye_tracker.track_right_eye(landmarks_2d)

            left_eyebrow = None
            right_eyebrow = None
            if self._config.enable_eyebrow_tracking:
                left_eyebrow = self._eyebrow_tracker.track_left_eyebrow(landmarks_2d)
                right_eyebrow = self._eyebrow_tracker.track_right_eyebrow(landmarks_2d)

            nose = None
            if self._config.enable_nose_tracking:
                nose = self._nose_tracker.track(landmarks_2d)

            mouth = None
            if self._config.enable_mouth_tracking:
                mouth = self._mouth_tracker.track(landmarks_2d)

            jaw = None
            if self._config.enable_jaw_tracking:
                jaw = self._jaw_tracker.track(landmarks_2d)

            head_pose = None
            if self._config.enable_head_pose:
                head_pose = self._pose_estimator.estimate(
                    landmarks_2d, landmarks_3d, width, height
                )

            face = FaceData(
                face_id=face_id,
                bounding_box=bbox,
                confidence=confidence,
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                left_eye=left_eye,
                right_eye=right_eye,
                left_eyebrow=left_eyebrow,
                right_eyebrow=right_eyebrow,
                nose=nose,
                mouth=mouth,
                jaw=jaw,
                head_pose=head_pose,
                expression=None,
            )

            if self._config.enable_expression_analysis:
                face.expression = self._expression_analyzer.analyze(face)

            if self._config.enable_gesture_detection:
                face.gestures = self._gesture_detector.detect(face)

            faces.append(face)

        elapsed = (time.perf_counter() - start_time) * 1000

        tracking_frame = TrackingFrame(
            faces=faces,
            frame_number=self._frame_count,
            timestamp=time.time(),
            processing_time_ms=elapsed,
            frame_width=width,
            frame_height=height,
        )

        self._fire_event("on_frame", tracking_frame)
        if tracking_frame.has_faces:
            self._fire_event("on_face_detected", tracking_frame)

        return tracking_frame

    def draw(self, frame: np.ndarray, tracking_frame: TrackingFrame) -> np.ndarray:
        return self._visualizer.draw_frame(frame, tracking_frame)

    def draw_dashboard(
        self, frame: np.ndarray, tracking_frame: TrackingFrame
    ) -> np.ndarray:
        return self._dashboard.draw_dashboard(frame, tracking_frame)

    def on_gesture(self, gesture_type: GestureType, callback: Callable):
        self._gesture_detector.on_gesture(gesture_type, callback)

    def on(self, event: str, callback: Callable):
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event] = [
                cb for cb in self._callbacks[event] if cb != callback
            ]

    def _fire_event(self, event: str, *args):
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args)
            except Exception:
                pass

    def run_camera(
        self,
        camera_index: int = 0,
        window_name: str = "Spectral",
        show_dashboard: bool = False,
        on_frame: Optional[Callable] = None,
    ):
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")

        self._running = True

        try:
            while self._running:
                ret, frame = self._capture.read()
                if not ret:
                    break

                tracking_frame = self.process_frame(frame)

                if on_frame:
                    on_frame(frame, tracking_frame)

                if show_dashboard:
                    display = self.draw_dashboard(frame, tracking_frame)
                else:
                    display = self.draw(frame, tracking_frame)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

        finally:
            self.stop()

    def run_video(
        self,
        video_path: str,
        window_name: str = "Spectral",
        show_dashboard: bool = False,
        on_frame: Optional[Callable] = None,
        output_path: Optional[str] = None,
    ):
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self._running = True

        try:
            while self._running:
                ret, frame = self._capture.read()
                if not ret:
                    break

                tracking_frame = self.process_frame(frame)

                if on_frame:
                    on_frame(frame, tracking_frame)

                if show_dashboard:
                    display = self.draw_dashboard(frame, tracking_frame)
                else:
                    display = self.draw(frame, tracking_frame)

                if writer:
                    writer.write(display)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

        finally:
            if writer:
                writer.release()
            self.stop()

    def process_image(self, image_path: str) -> TrackingFrame:
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        original_config = self._config.static_image_mode
        self._config.static_image_mode = True
        self._detector = FaceDetector(self._config)
        result = self.process_frame(frame)
        self._config.static_image_mode = original_config
        self._detector = FaceDetector(self._config)
        return result

    def stop(self):
        self._running = False
        if self._capture and self._capture.isOpened():
            self._capture.release()
        cv2.destroyAllWindows()

    def reset(self):
        self._frame_count = 0
        self._pose_estimator.reset()
        self._expression_analyzer.reset()
        self._gesture_detector.reset()

    def release(self):
        self.stop()
        self._detector.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def get_data_stream(self, camera_index: int = 0):
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")

        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                tracking_frame = self.process_frame(frame)
                yield frame, tracking_frame
        finally:
            capture.release()
