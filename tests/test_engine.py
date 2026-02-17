import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from spectral import SpectralEngine, TrackingConfig


class TestSpectralEngine(unittest.TestCase):
    def test_initialization(self):
        engine = SpectralEngine()
        self.assertIsNotNone(engine)
        self.assertEqual(engine.frame_count, 0)
        self.assertFalse(engine.is_running)

    def test_custom_config(self):
        config = TrackingConfig(max_faces=1, enable_eye_tracking=False)
        engine = SpectralEngine(config)
        self.assertEqual(engine.config.max_faces, 1)
        self.assertFalse(engine.config.enable_eye_tracking)

    def test_context_manager(self):
        with SpectralEngine() as engine:
            self.assertIsNotNone(engine)

    def test_event_registration(self):
        engine = SpectralEngine()
        called = []

        def callback(data):
            called.append(data)

        engine.on("on_frame", callback)
        engine.off("on_frame", callback)

    def test_reset(self):
        engine = SpectralEngine()
        engine.reset()
        self.assertEqual(engine.frame_count, 0)


if __name__ == "__main__":
    unittest.main()
