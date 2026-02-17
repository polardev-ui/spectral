import time
from typing import List, Optional, Dict, Callable
from collections import deque

from spectral.models import (
    FaceData,
    GestureEvent,
    GestureType,
    TrackingConfig,
)


class GestureDetector:
    def __init__(self, config: TrackingConfig):
        self._config = config
        self._cooldowns: Dict[GestureType, float] = {}
        self._state_history: deque = deque(maxlen=60)
        self._blink_state_left = False
        self._blink_state_right = False
        self._blink_start_left = 0.0
        self._blink_start_right = 0.0
        self._mouth_was_open = False
        self._mouth_open_start = 0.0
        self._head_yaw_history: deque = deque(maxlen=30)
        self._head_pitch_history: deque = deque(maxlen=30)
        self._callbacks: Dict[GestureType, List[Callable]] = {}

    def detect(self, face: FaceData) -> List[GestureEvent]:
        now = time.time()
        gestures = []

        self._state_history.append((now, face))

        gestures.extend(self._detect_blinks(face, now))
        gestures.extend(self._detect_mouth_gestures(face, now))
        gestures.extend(self._detect_eyebrow_gestures(face, now))
        gestures.extend(self._detect_head_gestures(face, now))
        gestures.extend(self._detect_gaze_gestures(face, now))

        filtered = []
        for g in gestures:
            last_time = self._cooldowns.get(g.gesture_type, 0.0)
            if now - last_time >= self._config.gesture_cooldown:
                self._cooldowns[g.gesture_type] = now
                filtered.append(g)
                self._fire_callbacks(g)

        return filtered

    def on_gesture(self, gesture_type: GestureType, callback: Callable):
        if gesture_type not in self._callbacks:
            self._callbacks[gesture_type] = []
        self._callbacks[gesture_type].append(callback)

    def remove_gesture_callback(
        self, gesture_type: GestureType, callback: Callable
    ):
        if gesture_type in self._callbacks:
            self._callbacks[gesture_type] = [
                cb for cb in self._callbacks[gesture_type] if cb != callback
            ]

    def _fire_callbacks(self, event: GestureEvent):
        callbacks = self._callbacks.get(event.gesture_type, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def _detect_blinks(
        self, face: FaceData, now: float
    ) -> List[GestureEvent]:
        gestures = []
        if not face.left_eye or not face.right_eye:
            return gestures

        left_closed = face.left_eye.is_closed
        right_closed = face.right_eye.is_closed

        if left_closed and not self._blink_state_left:
            self._blink_state_left = True
            self._blink_start_left = now
        elif not left_closed and self._blink_state_left:
            self._blink_state_left = False
            duration = now - self._blink_start_left
            if duration < 0.5:
                if not right_closed and not self._blink_state_right:
                    gestures.append(
                        GestureEvent(
                            gesture_type=GestureType.WINK_LEFT,
                            confidence=0.8,
                            timestamp=now,
                            duration=duration,
                        )
                    )

        if right_closed and not self._blink_state_right:
            self._blink_state_right = True
            self._blink_start_right = now
        elif not right_closed and self._blink_state_right:
            self._blink_state_right = False
            duration = now - self._blink_start_right
            if duration < 0.5:
                if not left_closed and not self._blink_state_left:
                    gestures.append(
                        GestureEvent(
                            gesture_type=GestureType.WINK_RIGHT,
                            confidence=0.8,
                            timestamp=now,
                            duration=duration,
                        )
                    )

        if left_closed and right_closed:
            if self._blink_state_left and self._blink_state_right:
                pass
        elif not left_closed and not right_closed:
            if (
                self._blink_start_left > 0
                and self._blink_start_right > 0
                and abs(self._blink_start_left - self._blink_start_right) < 0.1
            ):
                duration = now - min(self._blink_start_left, self._blink_start_right)
                if 0.05 < duration < 0.5:
                    gestures.append(
                        GestureEvent(
                            gesture_type=GestureType.BLINK_BOTH,
                            confidence=0.9,
                            timestamp=now,
                            duration=duration,
                        )
                    )

        return gestures

    def _detect_mouth_gestures(
        self, face: FaceData, now: float
    ) -> List[GestureEvent]:
        gestures = []
        if not face.mouth:
            return gestures

        is_open = face.mouth.is_open

        if is_open and not self._mouth_was_open:
            self._mouth_was_open = True
            self._mouth_open_start = now
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.MOUTH_OPEN,
                    confidence=0.85,
                    timestamp=now,
                )
            )
        elif not is_open and self._mouth_was_open:
            self._mouth_was_open = False
            duration = now - self._mouth_open_start
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.MOUTH_CLOSE,
                    confidence=0.85,
                    timestamp=now,
                    duration=duration,
                )
            )

        if face.mouth.smile_amount > self._config.smile_threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.SMILE,
                    confidence=face.mouth.smile_amount,
                    timestamp=now,
                    metadata={"amount": face.mouth.smile_amount},
                )
            )

        if face.mouth.smile_amount < 0.2:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.FROWN,
                    confidence=1.0 - face.mouth.smile_amount,
                    timestamp=now,
                )
            )

        if face.mouth.pucker_amount > 0.7:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.KISS,
                    confidence=face.mouth.pucker_amount,
                    timestamp=now,
                )
            )

        if face.jaw and face.jaw.openness > 0.7:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.JAW_DROP,
                    confidence=face.jaw.openness,
                    timestamp=now,
                )
            )

        if face.jaw and face.jaw.clench_amount > 0.7:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.JAW_CLENCH,
                    confidence=face.jaw.clench_amount,
                    timestamp=now,
                )
            )

        return gestures

    def _detect_eyebrow_gestures(
        self, face: FaceData, now: float
    ) -> List[GestureEvent]:
        gestures = []
        threshold = self._config.eyebrow_raise_threshold

        if face.left_eyebrow and face.left_eyebrow.raise_amount > threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.EYEBROW_RAISE_LEFT,
                    confidence=face.left_eyebrow.raise_amount,
                    timestamp=now,
                )
            )

        if face.right_eyebrow and face.right_eyebrow.raise_amount > threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.EYEBROW_RAISE_RIGHT,
                    confidence=face.right_eyebrow.raise_amount,
                    timestamp=now,
                )
            )

        if (
            face.left_eyebrow
            and face.right_eyebrow
            and face.left_eyebrow.raise_amount > threshold
            and face.right_eyebrow.raise_amount > threshold
        ):
            avg_raise = (
                face.left_eyebrow.raise_amount + face.right_eyebrow.raise_amount
            ) / 2
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.EYEBROW_RAISE_BOTH,
                    confidence=avg_raise,
                    timestamp=now,
                )
            )

        return gestures

    def _detect_head_gestures(
        self, face: FaceData, now: float
    ) -> List[GestureEvent]:
        gestures = []
        if not face.head_pose:
            return gestures

        self._head_yaw_history.append(face.head_pose.yaw)
        self._head_pitch_history.append(face.head_pose.pitch)

        if len(self._head_yaw_history) >= 10:
            yaw_values = list(self._head_yaw_history)[-10:]
            yaw_range = max(yaw_values) - min(yaw_values)
            if yaw_range > 25:
                direction_changes = 0
                for i in range(2, len(yaw_values)):
                    if (yaw_values[i] - yaw_values[i - 1]) * (
                        yaw_values[i - 1] - yaw_values[i - 2]
                    ) < 0:
                        direction_changes += 1
                if direction_changes >= 2:
                    gestures.append(
                        GestureEvent(
                            gesture_type=GestureType.HEAD_SHAKE,
                            confidence=min(1.0, yaw_range / 40),
                            timestamp=now,
                        )
                    )

        if len(self._head_pitch_history) >= 10:
            pitch_values = list(self._head_pitch_history)[-10:]
            pitch_range = max(pitch_values) - min(pitch_values)
            if pitch_range > 15:
                direction_changes = 0
                for i in range(2, len(pitch_values)):
                    if (pitch_values[i] - pitch_values[i - 1]) * (
                        pitch_values[i - 1] - pitch_values[i - 2]
                    ) < 0:
                        direction_changes += 1
                if direction_changes >= 1:
                    gestures.append(
                        GestureEvent(
                            gesture_type=GestureType.HEAD_NOD,
                            confidence=min(1.0, pitch_range / 25),
                            timestamp=now,
                        )
                    )

        if face.head_pose.roll > 15:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.HEAD_TILT_RIGHT,
                    confidence=min(1.0, abs(face.head_pose.roll) / 30),
                    timestamp=now,
                )
            )
        elif face.head_pose.roll < -15:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.HEAD_TILT_LEFT,
                    confidence=min(1.0, abs(face.head_pose.roll) / 30),
                    timestamp=now,
                )
            )

        return gestures

    def _detect_gaze_gestures(
        self, face: FaceData, now: float
    ) -> List[GestureEvent]:
        gestures = []
        gaze = None
        if face.left_eye and face.left_eye.gaze_direction:
            gaze = face.left_eye.gaze_direction
        elif face.right_eye and face.right_eye.gaze_direction:
            gaze = face.right_eye.gaze_direction

        if gaze is None:
            return gestures

        threshold = 0.4

        if gaze.x < -threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.LOOK_LEFT,
                    confidence=min(1.0, abs(gaze.x)),
                    timestamp=now,
                    metadata={"gaze_x": gaze.x, "gaze_y": gaze.y},
                )
            )
        elif gaze.x > threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.LOOK_RIGHT,
                    confidence=min(1.0, abs(gaze.x)),
                    timestamp=now,
                    metadata={"gaze_x": gaze.x, "gaze_y": gaze.y},
                )
            )

        if gaze.y < -threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.LOOK_UP,
                    confidence=min(1.0, abs(gaze.y)),
                    timestamp=now,
                    metadata={"gaze_x": gaze.x, "gaze_y": gaze.y},
                )
            )
        elif gaze.y > threshold:
            gestures.append(
                GestureEvent(
                    gesture_type=GestureType.LOOK_DOWN,
                    confidence=min(1.0, abs(gaze.y)),
                    timestamp=now,
                    metadata={"gaze_x": gaze.x, "gaze_y": gaze.y},
                )
            )

        return gestures

    def reset(self):
        self._cooldowns.clear()
        self._state_history.clear()
        self._blink_state_left = False
        self._blink_state_right = False
        self._blink_start_left = 0.0
        self._blink_start_right = 0.0
        self._mouth_was_open = False
        self._mouth_open_start = 0.0
        self._head_yaw_history.clear()
        self._head_pitch_history.clear()
