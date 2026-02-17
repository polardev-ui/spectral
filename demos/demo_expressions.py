import sys
import os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig
from spectral.models import Expression


EXPRESSION_COLORS = {
    Expression.NEUTRAL: (180, 180, 180),
    Expression.HAPPY: (0, 220, 100),
    Expression.SAD: (200, 100, 50),
    Expression.SURPRISED: (0, 200, 255),
    Expression.ANGRY: (0, 0, 230),
    Expression.DISGUSTED: (0, 150, 0),
    Expression.FEARFUL: (200, 0, 200),
    Expression.CONTEMPT: (100, 100, 200),
}


def main():
    config = TrackingConfig(
        max_faces=4,
        enable_expression_analysis=True,
        enable_gesture_detection=False,
        smoothing_factor=0.5,
    )

    engine = SpectralEngine(config)

    print("Spectral - Expression Analysis Demo")
    print("Press 'q' or ESC to quit")
    print("")
    print("Detectable expressions:")
    print("  - Neutral")
    print("  - Happy")
    print("  - Sad")
    print("  - Surprised")
    print("  - Angry")
    print("  - Disgusted")
    print("  - Fearful")
    print("  - Contempt")
    print("")

    try:
        for frame, tracking in engine.get_data_stream(camera_index=0):
            display = frame.copy()

            for face in tracking.faces:
                bbox = face.bounding_box
                expr = face.expression

                if not expr:
                    continue

                color = EXPRESSION_COLORS.get(expr.primary, (200, 200, 200))

                cv2.rectangle(
                    display,
                    bbox.top_left,
                    bbox.bottom_right,
                    color,
                    2,
                )

                label = f"{expr.primary.name}"
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )[0]
                cv2.rectangle(
                    display,
                    (bbox.x, bbox.y - label_size[1] - 14),
                    (bbox.x + label_size[0] + 10, bbox.y),
                    color,
                    -1,
                )
                cv2.putText(
                    display,
                    label,
                    (bbox.x + 5, bbox.y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                bar_x = bbox.x
                bar_y = bbox.y + bbox.height + 10
                bar_width = bbox.width
                bar_height = 12

                sorted_expressions = sorted(
                    expr.scores.items(), key=lambda x: x[1], reverse=True
                )

                for i, (exp, score) in enumerate(sorted_expressions[:4]):
                    y = bar_y + i * (bar_height + 4)
                    exp_color = EXPRESSION_COLORS.get(exp, (150, 150, 150))

                    cv2.rectangle(
                        display,
                        (bar_x, y),
                        (bar_x + bar_width, y + bar_height),
                        (40, 40, 40),
                        -1,
                    )

                    fill = int(bar_width * score)
                    cv2.rectangle(
                        display,
                        (bar_x, y),
                        (bar_x + fill, y + bar_height),
                        exp_color,
                        -1,
                    )

                    cv2.putText(
                        display,
                        f"{exp.name[:3]} {score:.0%}",
                        (bar_x + 3, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            h, w = display.shape[:2]
            fps = (
                1000.0 / tracking.processing_time_ms
                if tracking.processing_time_ms > 0
                else 0
            )
            cv2.putText(
                display,
                f"FPS: {fps:.0f} | Faces: {tracking.face_count}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Spectral - Expression Analysis", display)

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
