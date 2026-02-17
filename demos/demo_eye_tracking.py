import sys
import os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig


class GazeCanvas:
    def __init__(self, width=800, height=600):
        self._width = width
        self._height = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._cursor_x = width // 2
        self._cursor_y = height // 2
        self._trail = []
        self._max_trail = 200
        self._heatmap = np.zeros((height, width), dtype=np.float32)
        self._smoothing = 0.3

    def update(self, gaze_x, gaze_y):
        target_x = int(self._width / 2 + gaze_x * self._width / 2)
        target_y = int(self._height / 2 + gaze_y * self._height / 2)

        self._cursor_x = int(
            self._cursor_x * self._smoothing
            + target_x * (1 - self._smoothing)
        )
        self._cursor_y = int(
            self._cursor_y * self._smoothing
            + target_y * (1 - self._smoothing)
        )

        self._cursor_x = max(0, min(self._width - 1, self._cursor_x))
        self._cursor_y = max(0, min(self._height - 1, self._cursor_y))

        self._trail.append((self._cursor_x, self._cursor_y))
        if len(self._trail) > self._max_trail:
            self._trail.pop(0)

        cv2.circle(
            self._heatmap,
            (self._cursor_x, self._cursor_y),
            30,
            1.0,
            -1,
        )
        self._heatmap *= 0.998

    def render(self):
        self._canvas[:] = (20, 20, 20)

        heatmap_normalized = cv2.normalize(
            self._heatmap, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        mask = self._heatmap > 0.01
        self._canvas[mask] = heatmap_color[mask]

        for i in range(1, len(self._trail)):
            alpha = i / len(self._trail)
            color = (
                int(100 * alpha),
                int(255 * alpha),
                int(200 * alpha),
            )
            thickness = max(1, int(3 * alpha))
            cv2.line(
                self._canvas,
                self._trail[i - 1],
                self._trail[i],
                color,
                thickness,
                cv2.LINE_AA,
            )

        cv2.circle(
            self._canvas,
            (self._cursor_x, self._cursor_y),
            12,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            self._canvas,
            (self._cursor_x, self._cursor_y),
            3,
            (0, 255, 0),
            -1,
        )

        grid_spacing = 100
        for x in range(0, self._width, grid_spacing):
            cv2.line(
                self._canvas,
                (x, 0),
                (x, self._height),
                (40, 40, 40),
                1,
            )
        for y in range(0, self._height, grid_spacing):
            cv2.line(
                self._canvas,
                (0, y),
                (self._width, y),
                (40, 40, 40),
                1,
            )

        cv2.line(
            self._canvas,
            (self._width // 2, 0),
            (self._width // 2, self._height),
            (60, 60, 60),
            1,
        )
        cv2.line(
            self._canvas,
            (0, self._height // 2),
            (self._width, self._height // 2),
            (60, 60, 60),
            1,
        )

        cv2.putText(
            self._canvas,
            f"Cursor: ({self._cursor_x}, {self._cursor_y})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return self._canvas


def main():
    config = TrackingConfig(
        max_faces=1,
        enable_eye_tracking=True,
        enable_iris_tracking=True,
        enable_expression_analysis=False,
        enable_gesture_detection=False,
        enable_nose_tracking=False,
        enable_jaw_tracking=False,
        smoothing_factor=0.6,
    )

    engine = SpectralEngine(config)
    canvas = GazeCanvas(800, 600)

    print("Spectral - Eye Tracking and Gaze Demo")
    print("Press 'q' or ESC to quit")
    print("")
    print("Look around to see your gaze mapped to the canvas.")
    print("The heatmap shows where you look most frequently.")
    print("")

    try:
        for frame, tracking in engine.get_data_stream(camera_index=0):
            if tracking.has_faces:
                face = tracking.primary_face
                gaze_x = 0.0
                gaze_y = 0.0

                if face.left_eye and face.left_eye.gaze_direction:
                    gaze_x += face.left_eye.gaze_direction.x
                    gaze_y += face.left_eye.gaze_direction.y
                if face.right_eye and face.right_eye.gaze_direction:
                    gaze_x += face.right_eye.gaze_direction.x
                    gaze_y += face.right_eye.gaze_direction.y

                if face.left_eye and face.right_eye:
                    gaze_x /= 2
                    gaze_y /= 2

                canvas.update(gaze_x, gaze_y)

            gaze_display = canvas.render()

            camera_small = cv2.resize(frame, (200, 150))
            camera_overlay = engine.draw(camera_small, tracking)
            gaze_display[10 : 10 + 150, 590 : 590 + 200] = camera_overlay

            cv2.imshow("Spectral - Gaze Tracker", gaze_display)

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
