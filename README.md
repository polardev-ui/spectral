# Spectral

**A high-performance facial recognition and tracking engine built for developers.**

Spectral is an open-source Python library that provides real-time face detection, landmark tracking, expression analysis, gesture recognition, and gaze estimation. It is designed to be integrated into games, applications, creative tools, accessibility software, and any project that benefits from understanding human facial movement.

Built on top of MediaPipe and OpenCV, Spectral abstracts away the complexity of facial analysis and exposes a clean, intuitive API that developers can use with just a few lines of code.

Created by **Josh Clark**, a 14-year-old software engineer.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
  - [SpectralEngine](#spectralengine)
  - [TrackingConfig](#trackingconfig)
  - [TrackingFrame](#trackingframe)
  - [FaceData](#facedata)
  - [EyeData](#eyedata)
  - [EyebrowData](#eyebrowdata)
  - [NoseData](#nosedata)
  - [MouthData](#mouthdata)
  - [JawData](#jawdata)
  - [HeadPose](#headpose)
  - [ExpressionState](#expressionstate)
  - [GestureEvent](#gestureevent)
- [Integration Guide](#integration-guide)
  - [Using with Pygame](#using-with-pygame)
  - [Using with Flask / FastAPI](#using-with-flask--fastapi)
  - [Using with WebSocket Streaming](#using-with-websocket-streaming)
  - [Using with Unity (via Bridge)](#using-with-unity-via-bridge)
- [Demos](#demos)
- [Configuration](#configuration)
- [Data Structures](#data-structures)
- [Performance](#performance)
- [Testing](#testing)
- [License](#license)

---

## Features

**Face Detection and Tracking**
- Multi-face detection supporting up to 4 simultaneous faces
- 478-point facial landmark mesh (including iris landmarks)
- Bounding box detection with confidence scoring
- Persistent face identification across frames

**Eye Tracking**
- Individual left and right eye tracking
- Eye openness measurement and aspect ratio calculation
- Iris center detection and radius estimation
- Pupil dilation measurement
- Gaze direction estimation (horizontal and vertical)
- Blink detection with duration measurement

**Eyebrow Tracking**
- Independent left and right eyebrow tracking
- Raise amount quantification
- Furrow detection (inner brow compression)
- Arch height measurement

**Nose Tracking**
- Nose tip and bridge point detection
- Nostril position tracking (left and right)
- Nose width and length measurement
- Wrinkle detection

**Mouth Tracking**
- Lip landmark tracking (inner and outer contours)
- Mouth openness quantification
- Smile amount detection
- Pucker detection
- Corner position tracking
- Width and height measurement

**Jaw Tracking**
- Jaw contour landmark tracking
- Jaw openness measurement
- Lateral deviation detection
- Clench amount estimation

**Head Pose Estimation**
- 3D head orientation (pitch, yaw, roll)
- Position tracking in 3D space
- Forward-facing detection
- Tilt detection
- Rotation matrix and translation vector output

**Expression Analysis**
- Eight discrete expression categories: Neutral, Happy, Sad, Surprised, Angry, Disgusted, Fearful, Contempt
- Per-expression confidence scores
- Valence measurement (positive/negative emotional state)
- Arousal measurement (energy level)
- Expression intensity tracking
- Temporal smoothing for stable output

**Gesture Recognition**
- Blink detection (left, right, both)
- Wink detection (left, right)
- Smile and frown detection
- Mouth open/close events
- Eyebrow raise detection (left, right, both)
- Head nod and head shake detection
- Head tilt detection (left, right)
- Gaze direction gestures (look left, right, up, down)
- Kiss and jaw drop detection
- Event callback system for gesture-driven applications
- Configurable cooldown to prevent duplicate firing

**Visualization**
- Built-in rendering of all tracked features
- Full dashboard mode with metric panels
- Customizable colors and display options
- Performance overlay (FPS, processing time, face count)

**Utilities**
- Exponential smoothing filter
- One Euro filter for low-latency smoothing
- Moving average and weighted moving average
- Velocity tracking

---

## Requirements

- Python 3.9 or higher
- A webcam (for real-time demos)
- Operating System: macOS, Windows, or Linux

Dependencies are listed in `requirements.txt`:

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/polardev-ui/spectral.git
cd spectral
```

Install dependencies:

```bash
pip install -r requirements.txt
```

To install Spectral as a package in your environment:

```bash
pip install -e .
```

---

## Quick Start

The simplest way to get started is with the built-in camera loop:

```python
from spectral import SpectralEngine

engine = SpectralEngine()
engine.run_camera(show_dashboard=True)
```

For more control, process frames manually:

```python
import cv2
from spectral import SpectralEngine

engine = SpectralEngine()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracking = engine.process_frame(frame)

    if tracking.has_faces:
        face = tracking.primary_face
        print(f"Left eye openness: {face.left_eye.openness:.2f}")
        print(f"Smile amount: {face.mouth.smile_amount:.2f}")
        print(f"Head yaw: {face.head_pose.yaw:.1f}")

    display = engine.draw(frame, tracking)
    cv2.imshow("Spectral", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
engine.release()
```

---

## Architecture

```
spectral/
    __init__.py              Top-level exports
    engine.py                SpectralEngine (main entry point)
    models.py                All data structures and enums
    core/
        detector.py          Face detection and landmark extraction
        pose.py              Head pose estimation via PnP solving
        expression.py        Expression classification
        gestures.py          Gesture detection and event system
    tracking/
        eye_tracker.py       Eye and iris tracking
        eyebrow_tracker.py   Eyebrow tracking
        nose_tracker.py      Nose tracking
        mouth_tracker.py     Mouth tracking
        jaw_tracker.py       Jaw tracking
    utils/
        smoothing.py         Signal smoothing filters
        visualization.py     Rendering and dashboard
demos/
    demo_basic.py            Basic face tracking
    demo_dashboard.py        Full dashboard with all metrics
    demo_eye_tracking.py     Gaze tracking with heatmap
    demo_gestures.py         Gesture recognition with event log
    demo_expressions.py      Expression analysis with score bars
    demo_puppet.py           Virtual puppet controlled by face
    demo_head_pose.py        Head pose with gauges and compass
    demo_data_export.py      JSON export of tracking data
tests/
    test_models.py           Data structure tests
    test_smoothing.py        Smoothing filter tests
    test_engine.py           Engine initialization tests
```

---

## API Reference

### SpectralEngine

The main class that coordinates all tracking subsystems.

```python
from spectral import SpectralEngine, TrackingConfig

config = TrackingConfig(max_faces=2, smoothing_factor=0.4)
engine = SpectralEngine(config)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `process_frame(frame)` | Process a single BGR numpy frame. Returns a `TrackingFrame`. |
| `draw(frame, tracking_frame)` | Render tracking overlays onto the frame. Returns the annotated frame. |
| `draw_dashboard(frame, tracking_frame)` | Render full dashboard with metric panels. Returns the combined image. |
| `run_camera(camera_index, window_name, show_dashboard, on_frame)` | Start a blocking camera loop with built-in display. |
| `run_video(video_path, window_name, show_dashboard, on_frame, output_path)` | Process a video file. Optionally write output to a new file. |
| `process_image(image_path)` | Process a single image file. Returns a `TrackingFrame`. |
| `get_data_stream(camera_index)` | Generator that yields `(frame, tracking_frame)` tuples from camera. |
| `on_gesture(gesture_type, callback)` | Register a callback for a specific gesture type. |
| `on(event, callback)` | Register a callback for engine events (`on_frame`, `on_face_detected`). |
| `off(event, callback)` | Remove an event callback. |
| `stop()` | Stop the camera loop and close windows. |
| `reset()` | Reset all internal state (frame counter, filters, gesture history). |
| `release()` | Release all resources. |

**Context Manager:**

```python
with SpectralEngine() as engine:
    for frame, tracking in engine.get_data_stream():
        if tracking.has_faces:
            print(tracking.primary_face.expression.primary.name)
```

---

### TrackingConfig

Configuration dataclass that controls every aspect of Spectral's behavior.

```python
config = TrackingConfig(
    max_faces=4,
    detection_confidence=0.5,
    tracking_confidence=0.5,
    enable_eye_tracking=True,
    enable_eyebrow_tracking=True,
    enable_nose_tracking=True,
    enable_mouth_tracking=True,
    enable_jaw_tracking=True,
    enable_head_pose=True,
    enable_expression_analysis=True,
    enable_gesture_detection=True,
    enable_iris_tracking=True,
    smoothing_factor=0.5,
    gesture_cooldown=0.3,
    blink_threshold=0.21,
    mouth_open_threshold=0.35,
    smile_threshold=0.4,
    eyebrow_raise_threshold=0.28,
    static_image_mode=False,
    refine_landmarks=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_faces` | int | 4 | Maximum number of faces to track simultaneously. |
| `detection_confidence` | float | 0.5 | Minimum confidence for face detection (0.0 to 1.0). |
| `tracking_confidence` | float | 0.5 | Minimum confidence for landmark tracking (0.0 to 1.0). |
| `enable_eye_tracking` | bool | True | Enable eye openness, iris, and gaze tracking. |
| `enable_eyebrow_tracking` | bool | True | Enable eyebrow raise and furrow tracking. |
| `enable_nose_tracking` | bool | True | Enable nose landmark tracking. |
| `enable_mouth_tracking` | bool | True | Enable mouth tracking (openness, smile, pucker). |
| `enable_jaw_tracking` | bool | True | Enable jaw contour and openness tracking. |
| `enable_head_pose` | bool | True | Enable 3D head pose estimation. |
| `enable_expression_analysis` | bool | True | Enable expression classification. |
| `enable_gesture_detection` | bool | True | Enable gesture event detection. |
| `enable_iris_tracking` | bool | True | Enable iris center and gaze estimation. |
| `smoothing_factor` | float | 0.5 | Temporal smoothing strength (0.0 = none, 1.0 = maximum). |
| `gesture_cooldown` | float | 0.3 | Minimum seconds between repeated gesture events. |
| `blink_threshold` | float | 0.21 | Eye aspect ratio below which a blink is detected. |
| `mouth_open_threshold` | float | 0.35 | Mouth openness above which the mouth is considered open. |
| `smile_threshold` | float | 0.4 | Smile amount above which a smile gesture is fired. |
| `eyebrow_raise_threshold` | float | 0.28 | Eyebrow raise amount above which a raise gesture is fired. |
| `static_image_mode` | bool | False | Set to True when processing single images instead of video. |
| `refine_landmarks` | bool | True | Enable refined landmark detection (includes iris points). |

---

### TrackingFrame

Returned by `process_frame()`. Contains all tracking data for a single frame.

| Property | Type | Description |
|----------|------|-------------|
| `faces` | List[FaceData] | All detected faces in the frame. |
| `frame_number` | int | Sequential frame counter. |
| `timestamp` | float | Unix timestamp of processing. |
| `processing_time_ms` | float | Time taken to process the frame in milliseconds. |
| `frame_width` | int | Width of the input frame in pixels. |
| `frame_height` | int | Height of the input frame in pixels. |
| `face_count` | int | Number of detected faces. |
| `has_faces` | bool | Whether any faces were detected. |
| `primary_face` | FaceData or None | The largest face in the frame. |

---

### FaceData

Contains all tracking data for a single detected face.

| Field | Type | Description |
|-------|------|-------------|
| `face_id` | int | Identifier for this face within the frame. |
| `bounding_box` | BoundingBox | Face bounding rectangle. |
| `confidence` | float | Detection confidence (0.0 to 1.0). |
| `landmarks_2d` | List[Point2D] | 478 facial landmark points in 2D. |
| `landmarks_3d` | List[Point3D] | 478 facial landmark points in 3D. |
| `left_eye` | EyeData or None | Left eye tracking data. |
| `right_eye` | EyeData or None | Right eye tracking data. |
| `left_eyebrow` | EyebrowData or None | Left eyebrow tracking data. |
| `right_eyebrow` | EyebrowData or None | Right eyebrow tracking data. |
| `nose` | NoseData or None | Nose tracking data. |
| `mouth` | MouthData or None | Mouth tracking data. |
| `jaw` | JawData or None | Jaw tracking data. |
| `head_pose` | HeadPose or None | 3D head pose estimation. |
| `expression` | ExpressionState or None | Expression analysis results. |
| `gestures` | List[GestureEvent] | Gestures detected this frame. |

---

### EyeData

| Field | Type | Description |
|-------|------|-------------|
| `center` | Point2D | Center point of the eye region. |
| `landmarks` | List[Point2D] | Eye contour landmark points. |
| `iris_center` | Point2D or None | Center of the iris (requires iris tracking). |
| `iris_radius` | float | Estimated radius of the iris in pixels. |
| `openness` | float | How open the eye is (0.0 = closed, 1.0 = fully open). |
| `gaze_direction` | Point2D or None | Normalized gaze vector. x: -1 (left) to 1 (right), y: -1 (up) to 1 (down). |
| `pupil_dilation` | float | Estimated pupil dilation (0.0 to 1.0). |
| `aspect_ratio` | float | Eye aspect ratio (vertical/horizontal). |
| `is_closed` | bool | Whether the eye is detected as closed. |
| `is_open` | bool | Inverse of is_closed. |

---

### EyebrowData

| Field | Type | Description |
|-------|------|-------------|
| `landmarks` | List[Point2D] | Eyebrow contour landmark points. |
| `center` | Point2D | Center point of the eyebrow. |
| `raise_amount` | float | How raised the eyebrow is (0.0 to 1.0). |
| `furrow_amount` | float | Inner brow compression amount (0.0 to 1.0). |
| `arch_height` | float | Arch height relative to endpoints (0.0 to 1.0). |
| `inner_point` | Point2D | Inner endpoint of the eyebrow. |
| `outer_point` | Point2D | Outer endpoint of the eyebrow. |

---

### NoseData

| Field | Type | Description |
|-------|------|-------------|
| `tip` | Point2D | Nose tip position. |
| `bridge` | Point2D | Top of the nose bridge. |
| `landmarks` | List[Point2D] | Nose contour landmark points. |
| `nostril_left` | Point2D | Left nostril position. |
| `nostril_right` | Point2D | Right nostril position. |
| `width` | float | Distance between nostrils in pixels. |
| `length` | float | Distance from bridge to tip in pixels. |
| `wrinkle_amount` | float | Nose wrinkle intensity (0.0 to 1.0). |

---

### MouthData

| Field | Type | Description |
|-------|------|-------------|
| `center` | Point2D | Center point of the mouth. |
| `landmarks` | List[Point2D] | Mouth contour landmark points. |
| `upper_lip_center` | Point2D | Top center of the upper lip. |
| `lower_lip_center` | Point2D | Bottom center of the lower lip. |
| `left_corner` | Point2D | Left corner of the mouth. |
| `right_corner` | Point2D | Right corner of the mouth. |
| `openness` | float | How open the mouth is (0.0 to 1.0). |
| `width` | float | Mouth width in pixels. |
| `height` | float | Mouth height in pixels. |
| `smile_amount` | float | Smile intensity (0.0 to 1.0). |
| `pucker_amount` | float | Pucker intensity (0.0 to 1.0). |
| `aspect_ratio` | float | Height to width ratio. |
| `is_open` | bool | Whether the mouth is considered open. |
| `is_closed` | bool | Inverse of is_open. |

---

### JawData

| Field | Type | Description |
|-------|------|-------------|
| `landmarks` | List[Point2D] | Jaw contour landmark points. |
| `center` | Point2D | Center of the lower jaw region. |
| `openness` | float | Jaw openness (0.0 to 1.0). |
| `width` | float | Jaw width in pixels. |
| `deviation` | float | Lateral deviation (-1.0 left to 1.0 right). |
| `clench_amount` | float | Jaw clench intensity (0.0 to 1.0). |

---

### HeadPose

| Field | Type | Description |
|-------|------|-------------|
| `pitch` | float | Up/down rotation in degrees. Positive = looking down. |
| `yaw` | float | Left/right rotation in degrees. Positive = looking right. |
| `roll` | float | Tilt rotation in degrees. Positive = tilting right. |
| `position` | Point3D | Estimated 3D position. |
| `rotation_matrix` | List[List[float]] or None | 3x3 rotation matrix. |
| `translation_vector` | List[float] or None | 3-element translation vector. |
| `is_facing_forward` | bool | True if pitch and yaw are both within 15 degrees of center. |
| `is_tilted` | bool | True if roll exceeds 10 degrees. |

---

### ExpressionState

| Field | Type | Description |
|-------|------|-------------|
| `primary` | Expression | The dominant expression (NEUTRAL, HAPPY, SAD, SURPRISED, ANGRY, DISGUSTED, FEARFUL, CONTEMPT). |
| `confidence` | float | Confidence of the primary expression (0.0 to 1.0). |
| `scores` | Dict[Expression, float] | Normalized scores for all eight expressions. |
| `valence` | float | Emotional valence (-1.0 negative to 1.0 positive). |
| `arousal` | float | Emotional arousal (-1.0 calm to 1.0 excited). |
| `intensity` | float | Overall expression intensity (0.0 to 1.0). |

---

### GestureEvent

| Field | Type | Description |
|-------|------|-------------|
| `gesture_type` | GestureType | Type of gesture detected. |
| `confidence` | float | Detection confidence (0.0 to 1.0). |
| `timestamp` | float | Unix timestamp of detection. |
| `duration` | float | Duration in seconds (for blinks, mouth open/close). |
| `metadata` | Dict | Additional data (e.g., gaze coordinates for look gestures). |

**Available gesture types:**

| Gesture | Description |
|---------|-------------|
| BLINK_LEFT | Left eye blink |
| BLINK_RIGHT | Right eye blink |
| BLINK_BOTH | Both eyes blink simultaneously |
| WINK_LEFT | Left eye wink (right stays open) |
| WINK_RIGHT | Right eye wink (left stays open) |
| MOUTH_OPEN | Mouth opens |
| MOUTH_CLOSE | Mouth closes |
| SMILE | Smile detected |
| FROWN | Frown detected |
| EYEBROW_RAISE_LEFT | Left eyebrow raised |
| EYEBROW_RAISE_RIGHT | Right eyebrow raised |
| EYEBROW_RAISE_BOTH | Both eyebrows raised |
| JAW_DROP | Jaw opens wide |
| JAW_CLENCH | Jaw clenches |
| KISS | Lips puckered |
| HEAD_NOD | Vertical head motion |
| HEAD_SHAKE | Horizontal head motion |
| HEAD_TILT_LEFT | Head tilted left |
| HEAD_TILT_RIGHT | Head tilted right |
| LOOK_LEFT | Gaze directed left |
| LOOK_RIGHT | Gaze directed right |
| LOOK_UP | Gaze directed up |
| LOOK_DOWN | Gaze directed down |

---

## Integration Guide

### Using with Pygame

```python
import pygame
import cv2
import numpy as np
from spectral import SpectralEngine

pygame.init()
screen = pygame.display.set_mode((640, 480))
engine = SpectralEngine()
cap = cv2.VideoCapture(0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    tracking = engine.process_frame(frame)

    if tracking.has_faces:
        face = tracking.primary_face
        if face.left_eye and face.left_eye.gaze_direction:
            gaze = face.left_eye.gaze_direction
            cursor_x = int(320 + gaze.x * 200)
            cursor_y = int(240 + gaze.y * 200)
            pygame.draw.circle(screen, (0, 255, 0), (cursor_x, cursor_y), 10)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)
    surface = pygame.surfarray.make_surface(frame_rgb)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

cap.release()
engine.release()
pygame.quit()
```

### Using with Flask / FastAPI

```python
from fastapi import FastAPI, WebSocket
import cv2
import json
from spectral import SpectralEngine

app = FastAPI()
engine = SpectralEngine()

@app.websocket("/ws/tracking")
async def tracking_ws(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracking = engine.process_frame(frame)

            data = {
                "face_count": tracking.face_count,
                "faces": []
            }

            for face in tracking.faces:
                face_info = {
                    "smile": face.mouth.smile_amount if face.mouth else 0,
                    "left_eye_open": face.left_eye.openness if face.left_eye else 0,
                    "right_eye_open": face.right_eye.openness if face.right_eye else 0,
                    "expression": face.expression.primary.name if face.expression else "UNKNOWN",
                }
                data["faces"].append(face_info)

            await websocket.send_json(data)
    finally:
        cap.release()
```

### Using with WebSocket Streaming

```python
import asyncio
import websockets
import cv2
import json
from spectral import SpectralEngine

engine = SpectralEngine()

async def stream(websocket, path):
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracking = engine.process_frame(frame)

            if tracking.has_faces:
                face = tracking.primary_face
                payload = {
                    "head_pose": {
                        "pitch": face.head_pose.pitch,
                        "yaw": face.head_pose.yaw,
                        "roll": face.head_pose.roll,
                    } if face.head_pose else None,
                    "left_eye": {
                        "openness": face.left_eye.openness,
                        "gaze_x": face.left_eye.gaze_direction.x if face.left_eye.gaze_direction else 0,
                        "gaze_y": face.left_eye.gaze_direction.y if face.left_eye.gaze_direction else 0,
                    } if face.left_eye else None,
                    "mouth": {
                        "openness": face.mouth.openness,
                        "smile": face.mouth.smile_amount,
                    } if face.mouth else None,
                }
                await websocket.send(json.dumps(payload))

            await asyncio.sleep(0.033)
    finally:
        cap.release()

asyncio.run(websockets.serve(stream, "localhost", 8765))
```

### Using with Unity (via Bridge)

Spectral can serve as a Python-side face tracking backend for Unity by streaming data over a local TCP or WebSocket connection. Run Spectral as a server process, serialize tracking data to JSON, and read it in Unity with a C# WebSocket client.

**Python side:**

```python
import socket
import json
import cv2
from spectral import SpectralEngine

engine = SpectralEngine()
cap = cv2.VideoCapture(0)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 9000))
server.listen(1)
conn, addr = server.accept()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracking = engine.process_frame(frame)

    if tracking.has_faces:
        face = tracking.primary_face
        data = {
            "head_pitch": face.head_pose.pitch if face.head_pose else 0,
            "head_yaw": face.head_pose.yaw if face.head_pose else 0,
            "head_roll": face.head_pose.roll if face.head_pose else 0,
            "left_eye_open": face.left_eye.openness if face.left_eye else 1,
            "right_eye_open": face.right_eye.openness if face.right_eye else 1,
            "mouth_open": face.mouth.openness if face.mouth else 0,
            "smile": face.mouth.smile_amount if face.mouth else 0,
        }
        message = json.dumps(data) + "\n"
        conn.sendall(message.encode())

cap.release()
conn.close()
server.close()
```

---

## Demos

All demos are in the `demos/` directory and can be run directly:

| Demo | Command | Description |
|------|---------|-------------|
| Basic Tracking | `python demos/demo_basic.py` | Minimal face tracking with landmark overlay. |
| Dashboard | `python demos/demo_dashboard.py` | Full dashboard with real-time metric panels for all tracked features. |
| Eye Tracking | `python demos/demo_eye_tracking.py` | Gaze mapped to a canvas with trail visualization and heatmap. |
| Gesture Recognition | `python demos/demo_gestures.py` | Live gesture detection with event log (blinks, winks, smiles, nods). |
| Expression Analysis | `python demos/demo_expressions.py` | Expression classification with per-expression score bars. |
| Virtual Puppet | `python demos/demo_puppet.py` | A 2D puppet face that mirrors your facial movements in real time. |
| Head Pose | `python demos/demo_head_pose.py` | Head orientation with gauge bars and compass visualization. |
| Data Export | `python demos/demo_data_export.py` | Press 's' to save a full tracking snapshot to JSON. |

---

## Configuration

Spectral provides fine-grained control over which features are active and how they behave. Disabling features you do not need improves performance.

**Performance-optimized configuration (single face, minimal features):**

```python
config = TrackingConfig(
    max_faces=1,
    enable_eye_tracking=True,
    enable_eyebrow_tracking=False,
    enable_nose_tracking=False,
    enable_mouth_tracking=True,
    enable_jaw_tracking=False,
    enable_head_pose=False,
    enable_expression_analysis=False,
    enable_gesture_detection=False,
    smoothing_factor=0.3,
)
```

**Full-featured configuration (all features active):**

```python
config = TrackingConfig(
    max_faces=4,
    detection_confidence=0.6,
    tracking_confidence=0.6,
    enable_eye_tracking=True,
    enable_eyebrow_tracking=True,
    enable_nose_tracking=True,
    enable_mouth_tracking=True,
    enable_jaw_tracking=True,
    enable_head_pose=True,
    enable_expression_analysis=True,
    enable_gesture_detection=True,
    enable_iris_tracking=True,
    smoothing_factor=0.5,
    refine_landmarks=True,
)
```

---

## Data Structures

**Point2D** - A 2D coordinate with `x` and `y` fields. Includes `distance_to()`, `midpoint()`, `as_tuple()`, and `as_int_tuple()` methods.

**Point3D** - A 3D coordinate with `x`, `y`, and `z` fields. Includes `distance_to()`, `as_tuple()`, and `to_2d()` methods.

**BoundingBox** - A rectangle with `x`, `y`, `width`, and `height` fields. Includes `center`, `area`, `top_left`, `bottom_right`, `contains()`, and `overlap()` methods.

---

## Performance

Typical performance on a modern machine with all features enabled:

| Configuration | Faces | Approximate FPS |
|---------------|-------|-----------------|
| All features | 1 | 25-35 |
| All features | 4 | 15-25 |
| Eyes + Mouth only | 1 | 30-45 |
| Detection only | 1 | 40-60 |

Performance varies based on CPU, resolution, and the number of enabled features. Disabling features you do not need (head pose, expression analysis, gesture detection) will improve throughput.

---

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or with unittest:

```bash
python -m unittest discover tests/ -v
```

---

## License

Spectral is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Created by **Josh Clark**.
