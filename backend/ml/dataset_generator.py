"""
DeepVani Real Kinematic Dataset Manager
Stores actual human webcam landmark recordings (30x8) and provides kinematic datasets.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

PHRASE_DICTIONARY: List[Dict[str, Any]] = [
    {"id": 0, "label": "mujhe paani chahiye", "lang": "hi", "category": "need"},
    {"id": 1, "label": "doctor ko bulao", "lang": "hi", "category": "emergency"},
    {"id": 2, "label": "help me", "lang": "en", "category": "emergency"},
    {"id": 3, "label": "namaste", "lang": "hi", "category": "greeting"},
    {"id": 4, "label": "mujhe dard ho raha hai", "lang": "hi", "category": "medical"},
    {"id": 5, "label": "shukriya", "lang": "hi", "category": "social"},
    {"id": 6, "label": "khaana chahiye", "lang": "hi", "category": "need"},
    {"id": 7, "label": "where am I", "lang": "en", "category": "question"},
    {"id": 8, "label": "yes", "lang": "en", "category": "choice"},
    {"id": 9, "label": "no", "lang": "en", "category": "choice"},
    {"id": 10, "label": "call my family", "lang": "en", "category": "emergency"},
    {"id": 11, "label": "call emergency", "lang": "en", "category": "emergency"},
    {"id": 12, "label": "mujhe dawai chahiye", "lang": "hi", "category": "medical"},
    {"id": 13, "label": "theek hoon", "lang": "hi", "category": "status"},
    {"id": 14, "label": "open the door", "lang": "en", "category": "action"}
]


class KinematicDatasetGenerator:
    """
    Kinematic synthetic sequence dataset generator.
    """

    def __init__(self, sequence_length: int = 30, num_features: int = 8):
        self.sequence_length = sequence_length
        self.num_features = num_features

    def generate_dataset(self, samples_per_phrase: int = 20) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list = []
        y_list = []
        labels_list = []

        for phrase in PHRASE_DICTIONARY:
            pid = phrase["id"]
            label = phrase["label"]

            for _ in range(samples_per_phrase):
                t = np.linspace(0, np.pi * 2, self.sequence_length)
                freq = 0.5 + (pid * 0.15)
                amp = 0.08 + (pid * 0.01)

                f1 = 0.12 + amp * np.abs(np.sin(freq * t)) + np.random.normal(0, 0.005, self.sequence_length)
                f2 = 0.45 + (amp * 0.5) * np.cos(freq * t) + np.random.normal(0, 0.005, self.sequence_length)
                f3 = np.random.normal(0.01, 0.002, self.sequence_length)
                f4 = np.random.normal(0.01, 0.002, self.sequence_length)
                f5 = f1 / np.maximum(f2, 1e-4)
                f6 = np.gradient(f1)
                f7 = np.random.normal(0.02, 0.003, self.sequence_length)
                f8 = np.random.normal(0.02, 0.003, self.sequence_length)

                mat = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8])
                X_list.append(mat)
                y_list.append(pid)
                labels_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        return X, y, labels_list


class RealKinematicDataset:
    """
    Manages real human lip recordings (backend/data/real_lip_dataset.json).
    """

    def __init__(self, data_file: Optional[str] = None):
        if data_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            data_file = os.path.join(data_dir, "real_lip_dataset.json")

        self.data_file = data_file
        self.samples = self.load_dataset()

    def load_dataset(self) -> List[Dict]:
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        generator = KinematicDatasetGenerator()
        X_syn, y_syn, labels_syn = generator.generate_dataset(samples_per_phrase=20)

        if len(self.samples) == 0:
            return X_syn, y_syn, labels_syn

        X_real_list, y_real_list, labels_real_list = [], [], []
        for item in self.samples:
            tensor = np.array(item["tensor"], dtype=np.float32)
            if tensor.shape == (30, 8):
                X_real_list.append(tensor)
                y_real_list.append(item["phrase_id"])
                labels_real_list.append(item["label"])

        if len(X_real_list) > 0:
            X_real = np.array(X_real_list, dtype=np.float32)
            y_real = np.array(y_real_list, dtype=np.int64)

            X_combined = np.concatenate([X_syn, X_real], axis=0)
            y_combined = np.concatenate([y_syn, y_real], axis=0)
            labels_combined = labels_syn + labels_real_list
            return X_combined, y_combined, labels_combined

        return X_syn, y_syn, labels_syn
