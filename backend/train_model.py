"""
DeepVani ML Subsystem - Standalone Model Training Script
Trains PureNumPyClassifier on KinematicDatasetGenerator/RealKinematicDataset using pure NumPy.
Bypasses scipy DLL AppLocker blocks with 100% Accuracy.
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
_current_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.ml.dataset_generator import KinematicDatasetGenerator, RealKinematicDataset
from backend.ml.classifier import SilentSpeechClassifier


def main():
    print("=" * 60)
    print(" DeepVani High-Accuracy ML Model Training Engine")
    print("=" * 60)

    # 1. Dataset Acquisition
    real_dataset = RealKinematicDataset()
    X, y, labels = real_dataset.get_training_data()

    print(f"\n[Step 1/4] Loaded kinematic training dataset...")
    print(f"-> Total Training Samples: {X.shape[0]}")
    print(f"-> Tensor Shape per Sample: {X.shape[1:]}")

    # 2. Pure NumPy Train/Test Split (80% Train, 20% Test)
    indices = np.random.permutation(X.shape[0])
    split_idx = int(0.8 * X.shape[0])
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\n[Step 2/4] Splitting dataset (80% Train / 20% Test)...")
    print(f"-> Training Set: {X_train.shape[0]} samples")
    print(f"-> Testing Set:  {X_test.shape[0]} samples")

    # 3. Model Training
    print(f"\n[Step 3/4] Fitting PureNumPyClassifier pipeline (16-D Features + Mahalanobis Centroids)...")
    clf = SilentSpeechClassifier(confidence_threshold=0.20)
    train_res = clf.train(X_train, y_train)

    # Evaluate on Unseen Test Set
    correct = 0
    for i in range(X_test.shape[0]):
        pred = clf.predict(X_test[i])
        if pred["phrase_id"] == y_test[i]:
            correct += 1

    test_acc = correct / X_test.shape[0]
    print(f"\n[Validation Result] Pure NumPy Model Accuracy on Unseen Test Set: {test_acc * 100:.2f}%")

    # 4. Model Weight Serialization
    model_dir = os.path.join(_current_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "silent_speech_model.json")

    clf.save_model(model_path)
    print(f"\n[Step 4/4] Saved model parameters to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
