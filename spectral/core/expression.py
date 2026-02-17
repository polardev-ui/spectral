from typing import List, Optional, Dict

from spectral.models import (
    FaceData,
    ExpressionState,
    Expression,
    TrackingConfig,
)


class ExpressionAnalyzer:
    def __init__(self, config: TrackingConfig):
        self._config = config
        self._smoothing = config.smoothing_factor
        self._prev_scores = {}

    def analyze(self, face: FaceData) -> Optional[ExpressionState]:
        if not face.left_eye or not face.right_eye or not face.mouth:
            return None

        scores = self._compute_raw_scores(face)
        scores = self._smooth_scores(scores)
        scores = self._normalize_scores(scores)

        primary = max(scores, key=scores.get)
        confidence = scores[primary]
        intensity = max(scores.values())

        valence = self._compute_valence(scores)
        arousal = self._compute_arousal(scores)

        return ExpressionState(
            primary=primary,
            confidence=confidence,
            scores=scores,
            valence=valence,
            arousal=arousal,
            intensity=intensity,
        )

    def _compute_raw_scores(self, face: FaceData) -> Dict[Expression, float]:
        scores = {}

        eye_openness = 0.0
        if face.left_eye and face.right_eye:
            eye_openness = (face.left_eye.openness + face.right_eye.openness) / 2

        mouth_openness = face.mouth.openness if face.mouth else 0.0
        smile = face.mouth.smile_amount if face.mouth else 0.0
        pucker = face.mouth.pucker_amount if face.mouth else 0.0

        brow_raise = 0.0
        brow_furrow = 0.0
        if face.left_eyebrow and face.right_eyebrow:
            brow_raise = (
                face.left_eyebrow.raise_amount + face.right_eyebrow.raise_amount
            ) / 2
            brow_furrow = (
                face.left_eyebrow.furrow_amount + face.right_eyebrow.furrow_amount
            ) / 2

        jaw_open = face.jaw.openness if face.jaw else 0.0

        scores[Expression.NEUTRAL] = self._neutral_score(
            smile, brow_raise, brow_furrow, mouth_openness, eye_openness
        )

        scores[Expression.HAPPY] = self._happy_score(
            smile, brow_raise, eye_openness
        )

        scores[Expression.SAD] = self._sad_score(
            smile, brow_raise, brow_furrow, mouth_openness
        )

        scores[Expression.SURPRISED] = self._surprised_score(
            eye_openness, mouth_openness, brow_raise, jaw_open
        )

        scores[Expression.ANGRY] = self._angry_score(
            brow_furrow, smile, eye_openness, jaw_open
        )

        scores[Expression.DISGUSTED] = self._disgusted_score(
            brow_furrow, smile, mouth_openness
        )

        scores[Expression.FEARFUL] = self._fearful_score(
            eye_openness, brow_raise, mouth_openness, brow_furrow
        )

        scores[Expression.CONTEMPT] = self._contempt_score(face)

        return scores

    def _neutral_score(
        self,
        smile: float,
        brow_raise: float,
        brow_furrow: float,
        mouth_open: float,
        eye_open: float,
    ) -> float:
        deviation = (
            abs(smile - 0.4)
            + abs(brow_raise - 0.3)
            + abs(brow_furrow)
            + abs(mouth_open)
            + abs(eye_open - 0.6)
        )
        return max(0.0, 1.0 - deviation * 0.5)

    def _happy_score(
        self, smile: float, brow_raise: float, eye_open: float
    ) -> float:
        score = smile * 0.6 + max(0, brow_raise - 0.2) * 0.2 + eye_open * 0.2
        return max(0.0, min(1.0, score))

    def _sad_score(
        self,
        smile: float,
        brow_raise: float,
        brow_furrow: float,
        mouth_open: float,
    ) -> float:
        low_smile = max(0, 0.5 - smile)
        inner_raise = brow_furrow * 0.3
        score = low_smile * 0.5 + inner_raise + max(0, 0.3 - mouth_open) * 0.3
        return max(0.0, min(1.0, score))

    def _surprised_score(
        self,
        eye_open: float,
        mouth_open: float,
        brow_raise: float,
        jaw_open: float,
    ) -> float:
        score = (
            max(0, eye_open - 0.5) * 0.3
            + mouth_open * 0.3
            + max(0, brow_raise - 0.3) * 0.3
            + jaw_open * 0.1
        )
        return max(0.0, min(1.0, score))

    def _angry_score(
        self,
        brow_furrow: float,
        smile: float,
        eye_open: float,
        jaw_clench: float,
    ) -> float:
        low_smile = max(0, 0.4 - smile)
        narrow_eyes = max(0, 0.5 - eye_open)
        score = (
            brow_furrow * 0.4
            + low_smile * 0.2
            + narrow_eyes * 0.2
            + jaw_clench * 0.2
        )
        return max(0.0, min(1.0, score))

    def _disgusted_score(
        self, brow_furrow: float, smile: float, mouth_open: float
    ) -> float:
        score = brow_furrow * 0.4 + max(0, 0.3 - smile) * 0.3 + mouth_open * 0.3
        return max(0.0, min(1.0, score))

    def _fearful_score(
        self,
        eye_open: float,
        brow_raise: float,
        mouth_open: float,
        brow_furrow: float,
    ) -> float:
        score = (
            max(0, eye_open - 0.5) * 0.3
            + brow_raise * 0.3
            + mouth_open * 0.2
            + brow_furrow * 0.2
        )
        return max(0.0, min(1.0, score))

    def _contempt_score(self, face: FaceData) -> float:
        if not face.mouth:
            return 0.0
        left = face.mouth.left_corner
        right = face.mouth.right_corner
        center = face.mouth.center
        left_dist = abs(left.y - center.y)
        right_dist = abs(right.y - center.y)
        asymmetry = abs(left_dist - right_dist)
        face_width = face.bounding_box.width
        if face_width < 1e-6:
            return 0.0
        return max(0.0, min(1.0, (asymmetry / face_width) * 10))

    def _smooth_scores(
        self, scores: Dict[Expression, float]
    ) -> Dict[Expression, float]:
        if not self._prev_scores:
            self._prev_scores = scores.copy()
            return scores

        smoothed = {}
        alpha = self._smoothing
        for expr in scores:
            prev = self._prev_scores.get(expr, 0.0)
            smoothed[expr] = alpha * prev + (1 - alpha) * scores[expr]

        self._prev_scores = smoothed.copy()
        return smoothed

    def _normalize_scores(
        self, scores: Dict[Expression, float]
    ) -> Dict[Expression, float]:
        total = sum(scores.values())
        if total < 1e-6:
            return {expr: 1.0 / len(scores) for expr in scores}
        return {expr: val / total for expr, val in scores.items()}

    def _compute_valence(self, scores: Dict[Expression, float]) -> float:
        positive = scores.get(Expression.HAPPY, 0) + scores.get(Expression.SURPRISED, 0) * 0.3
        negative = (
            scores.get(Expression.SAD, 0)
            + scores.get(Expression.ANGRY, 0)
            + scores.get(Expression.DISGUSTED, 0)
            + scores.get(Expression.FEARFUL, 0)
        )
        return max(-1.0, min(1.0, positive - negative))

    def _compute_arousal(self, scores: Dict[Expression, float]) -> float:
        high_arousal = (
            scores.get(Expression.SURPRISED, 0)
            + scores.get(Expression.ANGRY, 0)
            + scores.get(Expression.FEARFUL, 0)
            + scores.get(Expression.HAPPY, 0) * 0.5
        )
        low_arousal = (
            scores.get(Expression.SAD, 0)
            + scores.get(Expression.NEUTRAL, 0)
        )
        return max(-1.0, min(1.0, high_arousal - low_arousal))

    def reset(self):
        self._prev_scores = {}
