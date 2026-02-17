__title__ = "Spectral"
__version__ = "1.0.0"
__author__ = "Josh Clark"
__description__ = "A high-performance facial recognition and tracking engine for developers."
__license__ = "MIT"

from spectral.models import (
    Point2D,
    Point3D,
    BoundingBox,
    EyeData,
    EyebrowData,
    NoseData,
    MouthData,
    JawData,
    HeadPose,
    FaceData,
    GestureEvent,
    ExpressionState,
    TrackingFrame,
    TrackingConfig,
)


def __getattr__(name):
    if name == "SpectralEngine":
        from spectral.engine import SpectralEngine
        return SpectralEngine
    raise AttributeError(f"module 'spectral' has no attribute {name}")


__all__ = [
    "SpectralEngine",
    "Point2D",
    "Point3D",
    "BoundingBox",
    "EyeData",
    "EyebrowData",
    "NoseData",
    "MouthData",
    "JawData",
    "HeadPose",
    "FaceData",
    "GestureEvent",
    "ExpressionState",
    "TrackingFrame",
    "TrackingConfig",
]
