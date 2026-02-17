import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig


def main():
    config = TrackingConfig(
        max_faces=2,
        enable_expression_analysis=True,
        enable_gesture_detection=True,
        enable_iris_tracking=True,
        smoothing_factor=0.4,
    )

    engine = SpectralEngine(config)

    print("Spectral - Full Dashboard Demo")
    print("Press 'q' or ESC to quit")
    print("")
    print("This demo shows a real-time dashboard with:")
    print("  - Eye tracking with iris and gaze")
    print("  - Eyebrow raise and furrow detection")
    print("  - Mouth openness, smile, and pucker metrics")
    print("  - Head pose estimation (pitch, yaw, roll)")
    print("  - Expression analysis with valence and arousal")
    print("")

    try:
        engine.run_camera(
            camera_index=0,
            window_name="Spectral - Dashboard",
            show_dashboard=True,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.release()


if __name__ == "__main__":
    main()
