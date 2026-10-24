"""
Unit tests for DeepVani KinematicBuffer and SilentSpeechClassifier.
"""

import os
import tempfile
import numpy as np
from backend.ml.classifier import KinematicBuffer, SilentSpeechClassifier
from backend.ml.dataset_generator import KinematicDatasetGenerator


def test_kinematic_buffer_push_and_tensor():
    buf = KinematicBuffer(buffer_size=30, feature_dim=8)
    assert not buf.is_full()

    for i in range(29):
        is_full = buf.push([0.1 * i] * 8)
        assert not is_full

    is_full = buf.push([0.5] * 8)
    assert is_full
    assert buf.is_full()

    tensor = buf.get_tensor()
    assert tensor.shape == (30, 8)


def test_classifier_training_and_prediction():
    gen = KinematicDatasetGenerator()
    X, y, labels = gen.generate_dataset(samples_per_phrase=10)

    clf = SilentSpeechClassifier(confidence_threshold=0.50)
    assert not clf.is_trained

    clf.train(X, y)
    assert clf.is_trained

    sample_tensor = X[0]
    result = clf.predict(sample_tensor)

    assert result["status"] == "success"
    assert "candidate" in result
    assert "confidence" in result
    assert "top_k" in result
    assert len(result["top_k"]) == 3


def test_classifier_quick_train_custom_phrase():
    gen = KinematicDatasetGenerator()
    X, y, labels = gen.generate_dataset(samples_per_phrase=5)

    clf = SilentSpeechClassifier(confidence_threshold=0.50)
    clf.train(X, y)
    initial_count = len(clf.phrase_dict)

    # Add custom phrase
    custom_samples = [np.ones((30, 8), dtype=np.float32) * 0.8 for _ in range(5)]
    success = clf.quick_train_phrase("Calling Nurse Immediately", custom_samples, "en")

    assert success
    assert len(clf.phrase_dict) == initial_count + 1

    # Predict with custom sample
    res = clf.predict(custom_samples[0])
    assert res["status"] == "success"
    assert res["raw_candidate"] == "Calling Nurse Immediately"


def test_classifier_serialization():
    gen = KinematicDatasetGenerator()
    X, y, labels = gen.generate_dataset(samples_per_phrase=5)

    clf = SilentSpeechClassifier()
    clf.train(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")
        clf.save_model(model_path)
        assert os.path.exists(model_path)

        new_clf = SilentSpeechClassifier()
        loaded = new_clf.load_model(model_path)
        assert loaded
        assert new_clf.is_trained
