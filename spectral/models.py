from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum, auto
import time


class Expression(Enum):
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    SURPRISED = auto()
    ANGRY = auto()
    DISGUSTED = auto()
    FEARFUL = auto()
    CONTEMPT = auto()


class GestureType(Enum):
    BLINK_LEFT = auto()
    BLINK_RIGHT = auto()
    BLINK_BOTH = auto()
    WINK_LEFT = auto()
    WINK_RIGHT = auto()
    MOUTH_OPEN = auto()
    MOUTH_CLOSE = auto()
    SMILE = auto()
    FROWN = auto()
    EYEBROW_RAISE_LEFT = auto()
    EYEBROW_RAISE_RIGHT = auto()
    EYEBROW_RAISE_BOTH = auto()
    JAW_DROP = auto()
    JAW_CLENCH = auto()
    KISS = auto()
    TONGUE_OUT = auto()
    HEAD_NOD = auto()
    HEAD_SHAKE = auto()
    HEAD_TILT_LEFT = auto()
    HEAD_TILT_RIGHT = auto()
    LOOK_LEFT = auto()
    LOOK_RIGHT = auto()
    LOOK_UP = auto()
    LOOK_DOWN = auto()


@dataclass
class Point2D:
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def midpoint(self, other: "Point2D") -> "Point2D":
        return Point2D((self.x + other.x) / 2, (self.y + other.y) / 2)

    def as_tuple(self) -> tuple:
        return (self.x, self.y)

    def as_int_tuple(self) -> tuple:
        return (int(self.x), int(self.y))


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def distance_to(self, other: "Point3D") -> float:
        return (
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        ) ** 0.5

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    def to_2d(self) -> Point2D:
        return Point2D(self.x, self.y)


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Point2D:
        return Point2D(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def top_left(self) -> tuple:
        return (self.x, self.y)

    @property
    def bottom_right(self) -> tuple:
        return (self.x + self.width, self.y + self.height)

    def contains(self, point: Point2D) -> bool:
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def overlap(self, other: "BoundingBox") -> float:
        x_overlap = max(
            0,
            min(self.x + self.width, other.x + other.width) - max(self.x, other.x),
        )
        y_overlap = max(
            0,
            min(self.y + self.height, other.y + other.height) - max(self.y, other.y),
        )
        intersection = x_overlap * y_overlap
        union = self.area + other.area - intersection
        if union == 0:
            return 0.0
        return intersection / union


@dataclass
class EyeData:
    center: Point2D
    landmarks: List[Point2D]
    iris_center: Optional[Point2D]
    iris_radius: float
    openness: float
    gaze_direction: Optional[Point2D]
    pupil_dilation: float
    aspect_ratio: float
    is_closed: bool

    @property
    def is_open(self) -> bool:
        return not self.is_closed


@dataclass
class EyebrowData:
    landmarks: List[Point2D]
    center: Point2D
    raise_amount: float
    furrow_amount: float
    arch_height: float
    inner_point: Point2D
    outer_point: Point2D


@dataclass
class NoseData:
    tip: Point2D
    bridge: Point2D
    landmarks: List[Point2D]
    nostril_left: Point2D
    nostril_right: Point2D
    width: float
    length: float
    wrinkle_amount: float


@dataclass
class MouthData:
    center: Point2D
    landmarks: List[Point2D]
    upper_lip_center: Point2D
    lower_lip_center: Point2D
    left_corner: Point2D
    right_corner: Point2D
    openness: float
    width: float
    height: float
    smile_amount: float
    pucker_amount: float
    aspect_ratio: float
    is_open: bool

    @property
    def is_closed(self) -> bool:
        return not self.is_open


@dataclass
class JawData:
    landmarks: List[Point2D]
    center: Point2D
    openness: float
    width: float
    deviation: float
    clench_amount: float


@dataclass
class HeadPose:
    pitch: float
    yaw: float
    roll: float
    position: Point3D
    rotation_matrix: Optional[List[List[float]]] = None
    translation_vector: Optional[List[float]] = None

    @property
    def is_facing_forward(self) -> bool:
        return abs(self.pitch) < 15 and abs(self.yaw) < 15

    @property
    def is_tilted(self) -> bool:
        return abs(self.roll) > 10


@dataclass
class ExpressionState:
    primary: Expression
    confidence: float
    scores: Dict[Expression, float]
    valence: float
    arousal: float
    intensity: float


@dataclass
class GestureEvent:
    gesture_type: GestureType
    confidence: float
    timestamp: float
    duration: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class FaceData:
    face_id: int
    bounding_box: BoundingBox
    confidence: float
    landmarks_2d: List[Point2D]
    landmarks_3d: List[Point3D]
    left_eye: Optional[EyeData]
    right_eye: Optional[EyeData]
    left_eyebrow: Optional[EyebrowData]
    right_eyebrow: Optional[EyebrowData]
    nose: Optional[NoseData]
    mouth: Optional[MouthData]
    jaw: Optional[JawData]
    head_pose: Optional[HeadPose]
    expression: Optional[ExpressionState]
    gestures: List[GestureEvent] = field(default_factory=list)

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks_2d)


@dataclass
class TrackingFrame:
    faces: List[FaceData]
    frame_number: int
    timestamp: float
    processing_time_ms: float
    frame_width: int
    frame_height: int

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0

    @property
    def primary_face(self) -> Optional[FaceData]:
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.bounding_box.area)


@dataclass
class TrackingConfig:
    max_faces: int = 4
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    enable_eye_tracking: bool = True
    enable_eyebrow_tracking: bool = True
    enable_nose_tracking: bool = True
    enable_mouth_tracking: bool = True
    enable_jaw_tracking: bool = True
    enable_head_pose: bool = True
    enable_expression_analysis: bool = True
    enable_gesture_detection: bool = True
    enable_iris_tracking: bool = True
    smoothing_factor: float = 0.5
    gesture_cooldown: float = 0.3
    blink_threshold: float = 0.21
    mouth_open_threshold: float = 0.35
    smile_threshold: float = 0.4
    eyebrow_raise_threshold: float = 0.28
    static_image_mode: bool = False
    refine_landmarks: bool = True
