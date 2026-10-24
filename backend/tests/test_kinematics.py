"""
Unit tests for DeepVani KinematicExtractor module.
"""

import pytest
import numpy as np
from backend.ml.kinematics import KinematicExtractor


def test_kinematic_extractor_initialization():
    extractor = KinematicExtractor()
    assert extractor.prev_f1 is None
    assert extractor.prev_f2 is None
    assert extractor.prev_f6 is None


def test_empty_landmarks_returns_zero_vector():
    extractor = KinematicExtractor()
    vector = extractor.extract_vector([])
    assert len(vector) == 8
    assert vector == [0.0] * 8


def test_insufficient_landmarks_returns_zero_vector():
    extractor = KinematicExtractor()
    short_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(100)]
    vector = extractor.extract_vector(short_landmarks)
    assert len(vector) == 8
    assert vector == [0.0] * 8


def test_valid_landmarks_metric_calculation():
    extractor = KinematicExtractor()
    mock_landmarks = [{'x': 0.5 + 0.001 * i, 'y': 0.5 + 0.001 * i, 'z': 0.0} for i in range(320)]
    
    # Frame 1
    v1 = extractor.extract_vector(mock_landmarks)
    assert len(v1) == 8
    assert v1[5] == 0.0  # Velocity delta f6 is 0 on first frame
    assert v1[6] == 0.0  # Velocity delta f7 is 0 on first frame
    assert v1[7] == 0.0  # Acceleration f8 is 0 on first frame

    # Frame 2 with subtle movement
    mock_landmarks[13] = {'x': 0.5, 'y': 0.40, 'z': 0.0}  # Shift lip
    v2 = extractor.extract_vector(mock_landmarks)
    assert len(v2) == 8
    assert isinstance(v2[0], float)
    assert isinstance(v2[1], float)
