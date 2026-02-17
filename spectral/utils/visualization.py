import cv2
import numpy as np
from typing import List, Optional, Tuple

from spectral.models import (
    Point2D,
    BoundingBox,
    FaceData,
    EyeData,
    MouthData,
    NoseData,
    EyebrowData,
    HeadPose,
    TrackingFrame,
    Expression,
)


class Visualizer:
    COLOR_FACE_BOX = (0, 255, 0)
    COLOR_LANDMARKS = (255, 255, 255)
    COLOR_EYE = (0, 200, 255)
    COLOR_IRIS = (255, 0, 255)
    COLOR_EYEBROW = (0, 255, 200)
    COLOR_NOSE = (255, 200, 0)
    COLOR_MOUTH = (0, 100, 255)
    COLOR_JAW = (200, 200, 200)
    COLOR_GAZE = (0, 0, 255)
    COLOR_POSE = (255, 100, 0)
    COLOR_TEXT_BG = (0, 0, 0)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, show_landmarks: bool = True, show_metrics: bool = True):
        self._show_landmarks = show_landmarks
        self._show_metrics = show_metrics

    def draw_frame(
        self, frame: np.ndarray, tracking_frame: TrackingFrame
    ) -> np.ndarray:
        output = frame.copy()

        for face in tracking_frame.faces:
            output = self.draw_face(output, face)

        if self._show_metrics:
            output = self._draw_performance(output, tracking_frame)

        return output

    def draw_face(self, frame: np.ndarray, face: FaceData) -> np.ndarray:
        output = frame.copy()

        self._draw_bounding_box(output, face.bounding_box, face.confidence)

        if self._show_landmarks:
            self._draw_landmarks(output, face)

        if face.left_eye:
            self._draw_eye(output, face.left_eye, "L")
        if face.right_eye:
            self._draw_eye(output, face.right_eye, "R")

        if face.left_eyebrow:
            self._draw_eyebrow(output, face.left_eyebrow)
        if face.right_eyebrow:
            self._draw_eyebrow(output, face.right_eyebrow)

        if face.nose:
            self._draw_nose(output, face.nose)

        if face.mouth:
            self._draw_mouth(output, face.mouth)

        if face.head_pose:
            self._draw_head_pose(output, face)

        if face.expression and self._show_metrics:
            self._draw_expression(output, face)

        if face.gestures and self._show_metrics:
            self._draw_gestures(output, face)

        return output

    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        confidence: float,
    ):
        cv2.rectangle(
            frame,
            bbox.top_left,
            bbox.bottom_right,
            self.COLOR_FACE_BOX,
            2,
        )
        label = f"{confidence:.0%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            frame,
            (bbox.x, bbox.y - label_size[1] - 10),
            (bbox.x + label_size[0] + 6, bbox.y),
            self.COLOR_TEXT_BG,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (bbox.x + 3, bbox.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    def _draw_landmarks(self, frame: np.ndarray, face: FaceData):
        for point in face.landmarks_2d:
            cv2.circle(
                frame, point.as_int_tuple(), 1, self.COLOR_LANDMARKS, -1
            )

    def _draw_eye(self, frame: np.ndarray, eye: EyeData, label: str):
        for i in range(len(eye.landmarks)):
            p1 = eye.landmarks[i].as_int_tuple()
            p2 = eye.landmarks[(i + 1) % len(eye.landmarks)].as_int_tuple()
            cv2.line(frame, p1, p2, self.COLOR_EYE, 1, cv2.LINE_AA)

        if eye.iris_center:
            center = eye.iris_center.as_int_tuple()
            radius = max(1, int(eye.iris_radius))
            cv2.circle(frame, center, radius, self.COLOR_IRIS, 1, cv2.LINE_AA)
            cv2.circle(frame, center, 2, self.COLOR_IRIS, -1)

        if eye.gaze_direction and eye.iris_center:
            start = eye.iris_center.as_int_tuple()
            end_x = int(eye.iris_center.x + eye.gaze_direction.x * 30)
            end_y = int(eye.iris_center.y + eye.gaze_direction.y * 30)
            cv2.arrowedLine(
                frame,
                start,
                (end_x, end_y),
                self.COLOR_GAZE,
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )

    def _draw_eyebrow(self, frame: np.ndarray, eyebrow: EyebrowData):
        for i in range(len(eyebrow.landmarks) - 1):
            p1 = eyebrow.landmarks[i].as_int_tuple()
            p2 = eyebrow.landmarks[i + 1].as_int_tuple()
            cv2.line(frame, p1, p2, self.COLOR_EYEBROW, 2, cv2.LINE_AA)

    def _draw_nose(self, frame: np.ndarray, nose: NoseData):
        cv2.circle(
            frame, nose.tip.as_int_tuple(), 3, self.COLOR_NOSE, -1
        )
        cv2.circle(
            frame, nose.bridge.as_int_tuple(), 2, self.COLOR_NOSE, -1
        )
        cv2.circle(
            frame, nose.nostril_left.as_int_tuple(), 2, self.COLOR_NOSE, -1
        )
        cv2.circle(
            frame, nose.nostril_right.as_int_tuple(), 2, self.COLOR_NOSE, -1
        )
        for i in range(len(nose.landmarks) - 1):
            p1 = nose.landmarks[i].as_int_tuple()
            p2 = nose.landmarks[i + 1].as_int_tuple()
            cv2.line(frame, p1, p2, self.COLOR_NOSE, 1, cv2.LINE_AA)

    def _draw_mouth(self, frame: np.ndarray, mouth: MouthData):
        for i in range(len(mouth.landmarks)):
            p1 = mouth.landmarks[i].as_int_tuple()
            p2 = mouth.landmarks[(i + 1) % len(mouth.landmarks)].as_int_tuple()
            cv2.line(frame, p1, p2, self.COLOR_MOUTH, 1, cv2.LINE_AA)

        cv2.circle(
            frame, mouth.left_corner.as_int_tuple(), 3, self.COLOR_MOUTH, -1
        )
        cv2.circle(
            frame, mouth.right_corner.as_int_tuple(), 3, self.COLOR_MOUTH, -1
        )

    def _draw_head_pose(self, frame: np.ndarray, face: FaceData):
        if not face.head_pose or not face.nose:
            return

        nose = face.nose.tip
        length = 60

        pitch_rad = np.radians(face.head_pose.pitch)
        yaw_rad = np.radians(face.head_pose.yaw)
        roll_rad = np.radians(face.head_pose.roll)

        x_axis = (
            int(nose.x + length * (np.cos(yaw_rad) * np.cos(roll_rad))),
            int(
                nose.y
                + length
                * (
                    np.cos(pitch_rad) * np.sin(roll_rad)
                    + np.cos(roll_rad) * np.sin(pitch_rad) * np.sin(yaw_rad)
                )
            ),
        )
        y_axis = (
            int(nose.x - length * np.cos(yaw_rad) * np.sin(roll_rad)),
            int(
                nose.y
                + length
                * (
                    np.cos(pitch_rad) * np.cos(roll_rad)
                    - np.sin(pitch_rad) * np.sin(yaw_rad) * np.sin(roll_rad)
                )
            ),
        )
        z_axis = (
            int(nose.x + length * np.sin(yaw_rad)),
            int(nose.y - length * np.cos(yaw_rad) * np.sin(pitch_rad)),
        )

        origin = nose.as_int_tuple()
        cv2.arrowedLine(
            frame, origin, x_axis, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2
        )
        cv2.arrowedLine(
            frame, origin, y_axis, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2
        )
        cv2.arrowedLine(
            frame, origin, z_axis, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.2
        )

    def _draw_expression(self, frame: np.ndarray, face: FaceData):
        if not face.expression:
            return

        x = face.bounding_box.x
        y = face.bounding_box.y + face.bounding_box.height + 20

        expr_name = face.expression.primary.name
        conf = face.expression.confidence
        text = f"{expr_name} ({conf:.0%})"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 5),
            (x + text_size[0] + 6, y + 5),
            self.COLOR_TEXT_BG,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x + 3, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    def _draw_gestures(self, frame: np.ndarray, face: FaceData):
        if not face.gestures:
            return

        x = face.bounding_box.x + face.bounding_box.width + 10
        y = face.bounding_box.y + 15

        for i, gesture in enumerate(face.gestures[:5]):
            text = f"{gesture.gesture_type.name}"
            cv2.putText(
                frame,
                text,
                (x, y + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_performance(
        self, frame: np.ndarray, tracking_frame: TrackingFrame
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        fps = (
            1000.0 / tracking_frame.processing_time_ms
            if tracking_frame.processing_time_ms > 0
            else 0
        )

        lines = [
            f"FPS: {fps:.1f}",
            f"Faces: {tracking_frame.face_count}",
            f"Time: {tracking_frame.processing_time_ms:.1f}ms",
        ]

        y_start = 20
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            x = w - text_size[0] - 10
            y = y_start + i * 22
            cv2.rectangle(
                frame,
                (x - 4, y - text_size[1] - 4),
                (x + text_size[0] + 4, y + 4),
                self.COLOR_TEXT_BG,
                -1,
            )
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

        return frame


class DashboardVisualizer(Visualizer):
    def draw_dashboard(
        self, frame: np.ndarray, tracking_frame: TrackingFrame
    ) -> np.ndarray:
        output = self.draw_frame(frame, tracking_frame)

        if not tracking_frame.has_faces:
            return output

        face = tracking_frame.primary_face
        if face is None:
            return output

        h, w = output.shape[:2]
        panel_width = 280
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        y_pos = 30
        y_pos = self._draw_panel_section(panel, "EYE TRACKING", y_pos)
        if face.left_eye:
            y_pos = self._draw_metric(
                panel, "Left Open", face.left_eye.openness, y_pos
            )
            y_pos = self._draw_metric(
                panel, "Left AR", face.left_eye.aspect_ratio, y_pos, max_val=0.5
            )
        if face.right_eye:
            y_pos = self._draw_metric(
                panel, "Right Open", face.right_eye.openness, y_pos
            )
            y_pos = self._draw_metric(
                panel, "Right AR", face.right_eye.aspect_ratio, y_pos, max_val=0.5
            )

        y_pos += 10
        y_pos = self._draw_panel_section(panel, "EYEBROW TRACKING", y_pos)
        if face.left_eyebrow:
            y_pos = self._draw_metric(
                panel, "L Raise", face.left_eyebrow.raise_amount, y_pos
            )
        if face.right_eyebrow:
            y_pos = self._draw_metric(
                panel, "R Raise", face.right_eyebrow.raise_amount, y_pos
            )
        if face.left_eyebrow:
            y_pos = self._draw_metric(
                panel, "Furrow", face.left_eyebrow.furrow_amount, y_pos
            )

        y_pos += 10
        y_pos = self._draw_panel_section(panel, "MOUTH TRACKING", y_pos)
        if face.mouth:
            y_pos = self._draw_metric(
                panel, "Openness", face.mouth.openness, y_pos
            )
            y_pos = self._draw_metric(
                panel, "Smile", face.mouth.smile_amount, y_pos
            )
            y_pos = self._draw_metric(
                panel, "Pucker", face.mouth.pucker_amount, y_pos
            )

        y_pos += 10
        y_pos = self._draw_panel_section(panel, "HEAD POSE", y_pos)
        if face.head_pose:
            y_pos = self._draw_angle(panel, "Pitch", face.head_pose.pitch, y_pos)
            y_pos = self._draw_angle(panel, "Yaw", face.head_pose.yaw, y_pos)
            y_pos = self._draw_angle(panel, "Roll", face.head_pose.roll, y_pos)

        y_pos += 10
        y_pos = self._draw_panel_section(panel, "EXPRESSION", y_pos)
        if face.expression:
            y_pos = self._draw_text(
                panel,
                f"{face.expression.primary.name} ({face.expression.confidence:.0%})",
                y_pos,
            )
            y_pos = self._draw_metric(
                panel, "Valence", (face.expression.valence + 1) / 2, y_pos
            )
            y_pos = self._draw_metric(
                panel, "Arousal", (face.expression.arousal + 1) / 2, y_pos
            )

        combined = np.hstack([output, panel])
        return combined

    def _draw_panel_section(
        self, panel: np.ndarray, title: str, y: int
    ) -> int:
        cv2.putText(
            panel,
            title,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 200, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.line(panel, (10, y + 5), (270, y + 5), (60, 60, 60), 1)
        return y + 22

    def _draw_metric(
        self,
        panel: np.ndarray,
        label: str,
        value: float,
        y: int,
        max_val: float = 1.0,
    ) -> int:
        cv2.putText(
            panel,
            label,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        bar_x = 100
        bar_w = 140
        bar_h = 10
        cv2.rectangle(
            panel,
            (bar_x, y - bar_h),
            (bar_x + bar_w, y),
            (50, 50, 50),
            -1,
        )

        fill_w = int(bar_w * min(1.0, value / max_val))
        color = self._value_color(value / max_val)
        cv2.rectangle(
            panel,
            (bar_x, y - bar_h),
            (bar_x + fill_w, y),
            color,
            -1,
        )

        val_text = f"{value:.2f}"
        cv2.putText(
            panel,
            val_text,
            (bar_x + bar_w + 5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        return y + 18

    def _draw_angle(
        self, panel: np.ndarray, label: str, value: float, y: int
    ) -> int:
        cv2.putText(
            panel,
            f"{label}: {value:.1f} deg",
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return y + 18

    def _draw_text(self, panel: np.ndarray, text: str, y: int) -> int:
        cv2.putText(
            panel,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return y + 18

    def _value_color(self, normalized: float) -> tuple:
        if normalized < 0.3:
            return (80, 180, 80)
        elif normalized < 0.7:
            return (80, 200, 255)
        else:
            return (80, 80, 255)
