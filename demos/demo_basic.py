import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig


def main():
    config = TrackingConfig(
        max_faces=2,
        enable_expression_analysis=True,
        enable_gesture_detection=True,
        smoothing_factor=0.4,
    )

    engine = SpectralEngine(config)

    print("Spectral - Basic Face Tracking Demo")
    print("Press 'q' or ESC to quit")
    print("")

    try:
        engine.run_camera(
            camera_index=0,
            window_name="Spectral - Basic Tracking",
            show_dashboard=False,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.release()


if __name__ == "__main__":
    main()
