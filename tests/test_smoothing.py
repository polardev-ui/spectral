import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from spectral.utils.smoothing import (
    ExponentialSmoothing,
    OneEuroFilter,
    MovingAverage,
    WeightedMovingAverage,
    VelocityTracker,
)


class TestExponentialSmoothing(unittest.TestCase):
    def test_initial_value(self):
        s = ExponentialSmoothing(alpha=0.5)
        result = s.update(10.0)
        self.assertAlmostEqual(result, 10.0)

    def test_smoothing(self):
        s = ExponentialSmoothing(alpha=0.5)
        s.update(10.0)
        result = s.update(20.0)
        self.assertAlmostEqual(result, 15.0)

    def test_reset(self):
        s = ExponentialSmoothing(alpha=0.5)
        s.update(10.0)
        s.reset()
        self.assertIsNone(s.value)


class TestMovingAverage(unittest.TestCase):
    def test_single_value(self):
        ma = MovingAverage(window_size=3)
        result = ma.update(10.0)
        self.assertAlmostEqual(result, 10.0)

    def test_average(self):
        ma = MovingAverage(window_size=3)
        ma.update(10.0)
        ma.update(20.0)
        result = ma.update(30.0)
        self.assertAlmostEqual(result, 20.0)

    def test_window(self):
        ma = MovingAverage(window_size=2)
        ma.update(10.0)
        ma.update(20.0)
        result = ma.update(30.0)
        self.assertAlmostEqual(result, 25.0)


class TestWeightedMovingAverage(unittest.TestCase):
    def test_weighted(self):
        wma = WeightedMovingAverage(window_size=3)
        wma.update(10.0)
        wma.update(20.0)
        result = wma.update(30.0)
        expected = (10 * 1 + 20 * 2 + 30 * 3) / 6
        self.assertAlmostEqual(result, expected)


class TestOneEuroFilter(unittest.TestCase):
    def test_initial(self):
        f = OneEuroFilter()
        result = f.update(5.0, 0.0)
        self.assertAlmostEqual(result, 5.0)

    def test_smoothing(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f.update(0.0, 0.0)
        result = f.update(10.0, 0.033)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 10.0)


class TestVelocityTracker(unittest.TestCase):
    def test_zero_velocity(self):
        vt = VelocityTracker()
        vt.update(5.0, 0.0)
        vt.update(5.0, 1.0)
        self.assertAlmostEqual(vt.velocity, 0.0)

    def test_constant_velocity(self):
        vt = VelocityTracker()
        vt.update(0.0, 0.0)
        vt.update(10.0, 1.0)
        self.assertAlmostEqual(vt.velocity, 10.0)


if __name__ == "__main__":
    unittest.main()
