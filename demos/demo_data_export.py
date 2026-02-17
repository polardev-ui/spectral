import sys
import os
import cv2
import numpy as np
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig


def main():
    config = TrackingConfig(
        max_faces=1,
        enable_expression_analysis=True,
        enable_gesture_detection=True,
        smoothing_factor=0.4,
    )

    engine = SpectralEngine(config)

    print("Spectral - Data Export Demo")
    print("Press 'q' or ESC to quit")
    print("Press 's' to save a snapshot to JSON")
    print("")

    snapshot_count = 0

    try:
        for frame, tracking in engine.get_data_stream(camera_index=0):
            display = engine.draw(frame, tracking)
            cv2.imshow("Spectral - Data Export", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            elif key == ord("s") and tracking.has_faces:
                snapshot_count += 1
                data = serialize_frame(tracking)
                filename = f"spectral_snapshot_{snapshot_count}.json"
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved: {filename}")

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.release()


def serialize_frame(tracking):
    data = {
        "frame_number": tracking.frame_number,
        "timestamp": tracking.timestamp,
        "processing_time_ms": tracking.processing_time_ms,
        "frame_width": tracking.frame_width,
        "frame_height": tracking.frame_height,
        "face_count": tracking.face_count,
        "faces": [],
    }

    for face in tracking.faces:
        face_data = {
            "face_id": face.face_id,
            "confidence": face.confidence,
            "bounding_box": {
                "x": face.bounding_box.x,
                "y": face.bounding_box.y,
                "width": face.bounding_box.width,
                "height": face.bounding_box.height,
            },
            "landmark_count": face.landmark_count,
        }

        if face.left_eye:
            face_data["left_eye"] = {
                "center": {"x": face.left_eye.center.x, "y": face.left_eye.center.y},
                "openness": face.left_eye.openness,
                "aspect_ratio": face.left_eye.aspect_ratio,
                "is_closed": face.left_eye.is_closed,
                "pupil_dilation": face.left_eye.pupil_dilation,
            }
            if face.left_eye.iris_center:
                face_data["left_eye"]["iris_center"] = {
                    "x": face.left_eye.iris_center.x,
                    "y": face.left_eye.iris_center.y,
                }
            if face.left_eye.gaze_direction:
                face_data["left_eye"]["gaze"] = {
                    "x": face.left_eye.gaze_direction.x,
                    "y": face.left_eye.gaze_direction.y,
                }

        if face.right_eye:
            face_data["right_eye"] = {
                "center": {"x": face.right_eye.center.x, "y": face.right_eye.center.y},
                "openness": face.right_eye.openness,
                "aspect_ratio": face.right_eye.aspect_ratio,
                "is_closed": face.right_eye.is_closed,
                "pupil_dilation": face.right_eye.pupil_dilation,
            }
            if face.right_eye.iris_center:
                face_data["right_eye"]["iris_center"] = {
                    "x": face.right_eye.iris_center.x,
                    "y": face.right_eye.iris_center.y,
                }
            if face.right_eye.gaze_direction:
                face_data["right_eye"]["gaze"] = {
                    "x": face.right_eye.gaze_direction.x,
                    "y": face.right_eye.gaze_direction.y,
                }

        if face.left_eyebrow:
            face_data["left_eyebrow"] = {
                "raise_amount": face.left_eyebrow.raise_amount,
                "furrow_amount": face.left_eyebrow.furrow_amount,
                "arch_height": face.left_eyebrow.arch_height,
            }

        if face.right_eyebrow:
            face_data["right_eyebrow"] = {
                "raise_amount": face.right_eyebrow.raise_amount,
                "furrow_amount": face.right_eyebrow.furrow_amount,
                "arch_height": face.right_eyebrow.arch_height,
            }

        if face.nose:
            face_data["nose"] = {
                "tip": {"x": face.nose.tip.x, "y": face.nose.tip.y},
                "bridge": {"x": face.nose.bridge.x, "y": face.nose.bridge.y},
                "width": face.nose.width,
                "length": face.nose.length,
                "wrinkle_amount": face.nose.wrinkle_amount,
            }

        if face.mouth:
            face_data["mouth"] = {
                "center": {"x": face.mouth.center.x, "y": face.mouth.center.y},
                "openness": face.mouth.openness,
                "width": face.mouth.width,
                "height": face.mouth.height,
                "smile_amount": face.mouth.smile_amount,
                "pucker_amount": face.mouth.pucker_amount,
                "is_open": face.mouth.is_open,
            }

        if face.jaw:
            face_data["jaw"] = {
                "openness": face.jaw.openness,
                "width": face.jaw.width,
                "deviation": face.jaw.deviation,
                "clench_amount": face.jaw.clench_amount,
            }

        if face.head_pose:
            face_data["head_pose"] = {
                "pitch": face.head_pose.pitch,
                "yaw": face.head_pose.yaw,
                "roll": face.head_pose.roll,
                "is_facing_forward": face.head_pose.is_facing_forward,
            }

        if face.expression:
            face_data["expression"] = {
                "primary": face.expression.primary.name,
                "confidence": face.expression.confidence,
                "valence": face.expression.valence,
                "arousal": face.expression.arousal,
                "intensity": face.expression.intensity,
                "scores": {
                    expr.name: score
                    for expr, score in face.expression.scores.items()
                },
            }

        if face.gestures:
            face_data["gestures"] = [
                {
                    "type": g.gesture_type.name,
                    "confidence": g.confidence,
                    "timestamp": g.timestamp,
                }
                for g in face.gestures
            ]

        data["faces"].append(face_data)

    return data


if __name__ == "__main__":
    main()
