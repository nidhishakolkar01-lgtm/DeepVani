"""
DeepVani ML Subsystem - PyTorch 1D-CNN Viseme Sequence Classifier
Provides a temporal 1D-Convolutional Neural Network option for viseme matrix tensor (30, 8) classification.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class VisemeCNN1D(nn.Module):
        """
        PyTorch 1D-CNN model architecture for 30x8 kinematic tensors.
        """
        def __init__(self, num_classes: int = 15):
            super(VisemeCNN1D, self).__init__()
            # Input: (Batch, 8, 30)
            self.conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=2)  # Output: (Batch, 32, 15)

            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.AdaptiveAvgPool1d(1)     # Output: (Batch, 64, 1)

            self.fc = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            # Input x: (Batch, 8, 30)
            x = self.pool1(self.relu(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            x = x.squeeze(-1)  # (Batch, 64)
            x = self.dropout(x)
            out = self.fc(x)   # (Batch, num_classes)
            return out


class PyTorchSilentSpeechClassifier:
    """
    PyTorch-backed classifier wrapper for DeepVani temporal tensor classification.
    """

    def __init__(self, num_classes: int = 15, confidence_threshold: float = 0.70):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.is_trained = False

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = VisemeCNN1D(num_classes=num_classes).to(self.device)
        else:
            self.device = None
            self.model = None

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 25, batch_size: int = 32, lr: float = 0.001):
        """
        Trains PyTorch 1D-CNN on tensor dataset X (N, 30, 8) and labels y (N,).
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not installed. PyTorchSilentSpeechClassifier disabled.")
            return

        # Transpose input X from (N, 30, 8) to (N, 8, 30) for Conv1d
        X_transposed = np.transpose(X, (0, 2, 1)).astype(np.float32)

        dataset = TensorDataset(torch.from_numpy(X_transposed), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        self.is_trained = True

    def predict(self, matrix_tensor: np.ndarray) -> Dict[str, Any]:
        """
        Runs PyTorch inference on single (30, 8) matrix tensor.
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            return {"status": "error", "message": "PyTorch model not trained or installed."}

        self.model.eval()
        with torch.no_grad():
            # Shape: (1, 8, 30)
            tensor_inp = torch.from_numpy(np.transpose(matrix_tensor, (1, 0)).astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.model(tensor_inp)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            max_idx = int(np.argmax(probs))
            max_conf = float(probs[max_idx])

            return {
                "status": "success",
                "phrase_id": max_idx,
                "confidence": round(max_conf, 4),
                "passed_gating": max_conf >= self.confidence_threshold,
                "probabilities": probs.tolist()
            }
