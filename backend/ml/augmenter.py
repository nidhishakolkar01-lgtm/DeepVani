"""
DeepVani ML Subsystem - Kinematic Data Augmentation Utility
Provides temporal time-warping, Gaussian noise injection, and amplitude scaling
to augment real-world webcam facial landmark datasets and prevent overfitting.
"""

import numpy as np
from typing import List, Tuple


class KinematicAugmenter:
    """
    Augments 30-frame temporal kinematic matrices of shape (30, 8).
    """

    def __init__(self, noise_std: float = 0.015, scale_range: Tuple[float, float] = (0.85, 1.15)):
        self.noise_std = noise_std
        self.scale_range = scale_range

    def add_gaussian_noise(self, tensor: np.ndarray) -> np.ndarray:
        """Adds Gaussian noise to metric features."""
        noise = np.random.normal(0, self.noise_std, tensor.shape).astype(np.float32)
        return tensor + noise

    def scale_amplitude(self, tensor: np.ndarray) -> np.ndarray:
        """Scales aperture and width metrics by a random scaling factor."""
        factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scaled = tensor.copy()
        scaled[:, 0] *= factor  # f1 (aperture)
        scaled[:, 1] *= factor  # f2 (width)
        return scaled

    def time_warp(self, tensor: np.ndarray) -> np.ndarray:
        """
        Applies linear temporal interpolation to warp gesture speed.
        """
        num_frames, num_features = tensor.shape
        orig_indices = np.linspace(0, num_frames - 1, num_frames)

        # Warp speed by changing sample rate
        warp_factor = np.random.uniform(0.8, 1.2)
        warped_indices = np.linspace(0, num_frames - 1, int(num_frames * warp_factor))

        warped_tensor = np.zeros((num_frames, num_features), dtype=np.float32)
        for col in range(num_features):
            interp_vals = np.interp(orig_indices, warped_indices, np.interp(warped_indices, orig_indices, tensor[:, col]))
            warped_tensor[:, col] = interp_vals

        return warped_tensor

    def augment_sample(self, tensor: np.ndarray) -> np.ndarray:
        """Applies a random combination of noise, scaling, and time warping."""
        augmented = tensor.copy()
        if np.random.rand() > 0.3:
            augmented = self.add_gaussian_noise(augmented)
        if np.random.rand() > 0.3:
            augmented = self.scale_amplitude(augmented)
        if np.random.rand() > 0.5:
            augmented = self.time_warp(augmented)
        return augmented

    def augment_dataset(self, X: np.ndarray, y: np.ndarray, num_augmented_copies: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates augmented copies for each sample in dataset X (N, 30, 8).
        """
        X_aug_list = [X]
        y_aug_list = [y]

        for _ in range(num_augmented_copies):
            augmented_samples = np.array([self.augment_sample(sample) for sample in X], dtype=np.float32)
            X_aug_list.append(augmented_samples)
            y_aug_list.append(y.copy())

        X_final = np.vstack(X_aug_list)
        y_final = np.concatenate(y_aug_list)

        return X_final, y_final
