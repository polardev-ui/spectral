import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from spectral.models import (
    Point2D,
    Point3D,
    BoundingBox,
    TrackingConfig,
    Expression,
    GestureType,
)


class TestPoint2D(unittest.TestCase):
    def test_distance(self):
        a = Point2D(0, 0)
        b = Point2D(3, 4)
        self.assertAlmostEqual(a.distance_to(b), 5.0)

    def test_midpoint(self):
        a = Point2D(0, 0)
        b = Point2D(10, 10)
        mid = a.midpoint(b)
        self.assertAlmostEqual(mid.x, 5.0)
        self.assertAlmostEqual(mid.y, 5.0)

    def test_as_tuple(self):
        p = Point2D(1.5, 2.5)
        self.assertEqual(p.as_tuple(), (1.5, 2.5))

    def test_as_int_tuple(self):
        p = Point2D(1.7, 2.3)
        self.assertEqual(p.as_int_tuple(), (1, 2))


class TestPoint3D(unittest.TestCase):
    def test_distance(self):
        a = Point3D(0, 0, 0)
        b = Point3D(1, 2, 2)
        self.assertAlmostEqual(a.distance_to(b), 3.0)

    def test_to_2d(self):
        p = Point3D(1.0, 2.0, 3.0)
        p2 = p.to_2d()
        self.assertAlmostEqual(p2.x, 1.0)
        self.assertAlmostEqual(p2.y, 2.0)


class TestBoundingBox(unittest.TestCase):
    def test_center(self):
        bb = BoundingBox(0, 0, 100, 100)
        center = bb.center
        self.assertAlmostEqual(center.x, 50.0)
        self.assertAlmostEqual(center.y, 50.0)

    def test_area(self):
        bb = BoundingBox(0, 0, 10, 20)
        self.assertEqual(bb.area, 200)

    def test_contains(self):
        bb = BoundingBox(10, 10, 100, 100)
        self.assertTrue(bb.contains(Point2D(50, 50)))
        self.assertFalse(bb.contains(Point2D(5, 5)))

    def test_overlap(self):
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 100, 100)
        iou = a.overlap(b)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)

    def test_no_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(100, 100, 10, 10)
        self.assertAlmostEqual(a.overlap(b), 0.0)


class TestTrackingConfig(unittest.TestCase):
    def test_defaults(self):
        config = TrackingConfig()
        self.assertEqual(config.max_faces, 4)
        self.assertTrue(config.enable_eye_tracking)
        self.assertTrue(config.enable_mouth_tracking)
        self.assertTrue(config.enable_expression_analysis)

    def test_custom(self):
        config = TrackingConfig(max_faces=1, enable_eye_tracking=False)
        self.assertEqual(config.max_faces, 1)
        self.assertFalse(config.enable_eye_tracking)


class TestEnums(unittest.TestCase):
    def test_expressions(self):
        self.assertEqual(len(Expression), 8)
        self.assertIn(Expression.NEUTRAL, Expression)
        self.assertIn(Expression.HAPPY, Expression)

    def test_gestures(self):
        self.assertGreater(len(GestureType), 20)
        self.assertIn(GestureType.BLINK_BOTH, GestureType)
        self.assertIn(GestureType.SMILE, GestureType)


if __name__ == "__main__":
    unittest.main()
