import sys
import os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig


def main():
    config = TrackingConfig(
        max_faces=1,
        enable_head_pose=True,
        enable_eye_tracking=True,
        enable_mouth_tracking=True,
        enable_eyebrow_tracking=True,
        enable_expression_analysis=False,
        enable_gesture_detection=False,
        smoothing_factor=0.5,
    )

    engine = SpectralEngine(config)

    print("Spectral - Head Pose Estimation Demo")
    print("Press 'q' or ESC to quit")
    print("")
    print("Turn your head to see pitch, yaw, and roll axes rendered.")
    print("The gauge panel shows real-time angle values.")
    print("")

    try:
        for frame, tracking in engine.get_data_stream(camera_index=0):
            display = engine.draw(frame, tracking)

            if tracking.has_faces:
                face = tracking.primary_face
                if face and face.head_pose:
                    display = draw_pose_panel(display, face.head_pose)

            cv2.imshow("Spectral - Head Pose", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.release()


def draw_pose_panel(frame, head_pose):
    h, w = frame.shape[:2]
    panel_w = 220
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 30)

    y = 30
    cv2.putText(
        panel,
        "HEAD POSE",
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 200, 255),
        1,
        cv2.LINE_AA,
    )
    y += 15
    cv2.line(panel, (15, y), (panel_w - 15, y), (60, 60, 60), 1)
    y += 30

    y = draw_gauge(panel, "PITCH", head_pose.pitch, y, (0, 200, 255))
    y += 15
    y = draw_gauge(panel, "YAW", head_pose.yaw, y, (0, 255, 100))
    y += 15
    y = draw_gauge(panel, "ROLL", head_pose.roll, y, (255, 100, 100))

    y += 30
    cv2.putText(
        panel,
        "STATUS",
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 200, 255),
        1,
        cv2.LINE_AA,
    )
    y += 25

    if head_pose.is_facing_forward:
        status = "Facing Forward"
        color = (0, 220, 0)
    else:
        status = "Turned Away"
        color = (0, 100, 220)

    cv2.putText(
        panel,
        status,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    y += 20

    if head_pose.is_tilted:
        cv2.putText(
            panel,
            "Head Tilted",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 0),
            1,
            cv2.LINE_AA,
        )
    y += 30

    y = draw_compass(panel, head_pose.yaw, head_pose.pitch, y)

    return np.hstack([frame, panel])


def draw_gauge(panel, label, value, y, color):
    cv2.putText(
        panel,
        f"{label}: {value:.1f}",
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    y += 15

    bar_x = 15
    bar_w = 190
    bar_h = 14
    center_x = bar_x + bar_w // 2

    cv2.rectangle(
        panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (50, 50, 50), -1
    )

    cv2.line(panel, (center_x, y), (center_x, y + bar_h), (80, 80, 80), 1)

    normalized = max(-1.0, min(1.0, value / 45.0))
    indicator_x = center_x + int(normalized * bar_w // 2)
    cv2.rectangle(
        panel,
        (min(center_x, indicator_x), y),
        (max(center_x, indicator_x), y + bar_h),
        color,
        -1,
    )

    cv2.circle(panel, (indicator_x, y + bar_h // 2), 5, color, -1, cv2.LINE_AA)

    return y + bar_h + 5


def draw_compass(panel, yaw, pitch, y):
    cx = 110
    cy = y + 60
    radius = 50

    cv2.circle(panel, (cx, cy), radius, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy), 3, (100, 100, 100), -1)

    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        rad = np.radians(angle - 90)
        lx = int(cx + (radius + 12) * np.cos(rad))
        ly = int(cy + (radius + 12) * np.sin(rad))
        cv2.putText(
            panel,
            label,
            (lx - 4, ly + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

    norm_yaw = max(-1.0, min(1.0, yaw / 45.0))
    norm_pitch = max(-1.0, min(1.0, pitch / 45.0))
    dot_x = int(cx + norm_yaw * radius * 0.8)
    dot_y = int(cy + norm_pitch * radius * 0.8)

    cv2.circle(panel, (dot_x, dot_y), 6, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.line(panel, (cx, cy), (dot_x, dot_y), (0, 150, 200), 1, cv2.LINE_AA)

    return y + 130


if __name__ == "__main__":
    main()
