import sys
import os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig
from spectral.models import GestureType


class GestureLogger:
    def __init__(self):
        self._log = []
        self._max_log = 15

    def add(self, gesture_name, confidence):
        self._log.append((gesture_name, confidence))
        if len(self._log) > self._max_log:
            self._log.pop(0)

    @property
    def entries(self):
        return self._log


def main():
    config = TrackingConfig(
        max_faces=1,
        enable_expression_analysis=True,
        enable_gesture_detection=True,
        gesture_cooldown=0.5,
        blink_threshold=0.2,
        smile_threshold=0.45,
        smoothing_factor=0.4,
    )

    engine = SpectralEngine(config)
    gesture_log = GestureLogger()

    def on_blink(event):
        gesture_log.add("BLINK", event.confidence)

    def on_wink_left(event):
        gesture_log.add("WINK LEFT", event.confidence)

    def on_wink_right(event):
        gesture_log.add("WINK RIGHT", event.confidence)

    def on_smile(event):
        gesture_log.add("SMILE", event.confidence)

    def on_mouth_open(event):
        gesture_log.add("MOUTH OPEN", event.confidence)

    def on_eyebrow_raise(event):
        gesture_log.add("EYEBROW RAISE", event.confidence)

    def on_head_nod(event):
        gesture_log.add("HEAD NOD", event.confidence)

    def on_head_shake(event):
        gesture_log.add("HEAD SHAKE", event.confidence)

    def on_look_left(event):
        gesture_log.add("LOOK LEFT", event.confidence)

    def on_look_right(event):
        gesture_log.add("LOOK RIGHT", event.confidence)

    engine.on_gesture(GestureType.BLINK_BOTH, on_blink)
    engine.on_gesture(GestureType.WINK_LEFT, on_wink_left)
    engine.on_gesture(GestureType.WINK_RIGHT, on_wink_right)
    engine.on_gesture(GestureType.SMILE, on_smile)
    engine.on_gesture(GestureType.MOUTH_OPEN, on_mouth_open)
    engine.on_gesture(GestureType.EYEBROW_RAISE_BOTH, on_eyebrow_raise)
    engine.on_gesture(GestureType.HEAD_NOD, on_head_nod)
    engine.on_gesture(GestureType.HEAD_SHAKE, on_head_shake)
    engine.on_gesture(GestureType.LOOK_LEFT, on_look_left)
    engine.on_gesture(GestureType.LOOK_RIGHT, on_look_right)

    print("Spectral - Gesture Recognition Demo")
    print("Press 'q' or ESC to quit")
    print("")
    print("Supported gestures:")
    print("  - Blink (both eyes)")
    print("  - Wink (left or right)")
    print("  - Smile")
    print("  - Mouth open")
    print("  - Eyebrow raise")
    print("  - Head nod and shake")
    print("  - Look left / right")
    print("")

    try:
        for frame, tracking in engine.get_data_stream(camera_index=0):
            display = engine.draw(frame, tracking)

            h, w = display.shape[:2]
            panel_height = 300
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            panel[:] = (25, 25, 25)

            cv2.putText(
                panel,
                "GESTURE LOG",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 200, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(panel, (15, 35), (w - 15, 35), (60, 60, 60), 1)

            for i, (name, conf) in enumerate(reversed(gesture_log.entries)):
                y = 55 + i * 18
                alpha = 1.0 - (i * 0.06)
                color = (
                    int(100 * alpha),
                    int(220 * alpha),
                    int(180 * alpha),
                )
                cv2.putText(
                    panel,
                    f"{name} ({conf:.0%})",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            combined = np.vstack([display, panel])
            cv2.imshow("Spectral - Gesture Recognition", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.release()


if __name__ == "__main__":
    main()
