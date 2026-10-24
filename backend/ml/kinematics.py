"""
DeepVani ML Subsystem - Kinematic Metric Extraction Module
Computes 8-dimensional normalized kinematic lip vectors (V_t) from 3D facial landmark coordinates.
Includes 3D Head Pose Estimation (Yaw/Pitch) & Adaptive Profile (Side-View) Compensation.
"""

import math
from typing import Dict, List, Tuple, Optional, Any


class KinematicExtractor:
    """
    Extracts 8-dimensional normalized kinematic lip metric vectors (V_t) 
    from MediaPipe 3D face mesh landmark dictionaries or arrays.
    Includes adaptive side-view (profile) pose compensation.
    
    Landmark Indices (MediaPipe Face Mesh):
    - Nose Tip: 1
    - Chin: 152
    - Face Width Anchors: 33 (Right), 263 (Left)
    - Lip Outer Corners: 61 (Right Corner), 291 (Left Corner)
    - Upper Lip Outer Center: 0
    - Upper Lip Outer Mids: 37 (Right), 267 (Left)
    - Inner Lips (Aperture): 13 (Upper Inner), 14 (Lower Inner)
    - Lower Lip Outer Center: 17
    - Lower Lip Outer Mids: 84 (Right), 314 (Left)
    """

    NOSE_TIP = 1
    CHIN = 152
    FACE_RIGHT = 33
    FACE_LEFT = 263
    LIP_CORNER_RIGHT = 61
    LIP_CORNER_LEFT = 291
    UPPER_LIP_CENTER = 0
    UPPER_LIP_MID_RIGHT = 37
    UPPER_LIP_MID_LEFT = 267
    UPPER_LIP_INNER = 13
    LOWER_LIP_INNER = 14
    LOWER_LIP_CENTER = 17
    LOWER_LIP_MID_RIGHT = 84
    LOWER_LIP_MID_LEFT = 314

    def __init__(self):
        # State tracking for velocity & acceleration deltas across consecutive frames
        self.prev_f1: Optional[float] = None
        self.prev_f2: Optional[float] = None
        self.prev_f6: Optional[float] = None

    def reset_state(self):
        """Reset temporal delta state tracking."""
        self.prev_f1 = None
        self.prev_f2 = None
        self.prev_f6 = None

    @staticmethod
    def _euclidean_distance_2d(p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Calculate 2D Euclidean distance between two landmark points."""
        dx = p1.get('x', 0.0) - p2.get('x', 0.0)
        dy = p1.get('y', 0.0) - p2.get('y', 0.0)
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def estimate_head_pose(landmarks: List[Dict[str, float]]) -> Tuple[float, float]:
        """
        Estimates 3D Head Pose (Yaw, Pitch) in degrees from facial landmarks.
        Yaw: Negative = Turning Left, Positive = Turning Right.
        """
        if not landmarks or len(landmarks) < 264:
            return 0.0, 0.0

        p1 = landmarks[KinematicExtractor.NOSE_TIP]
        p33 = landmarks[KinematicExtractor.FACE_RIGHT]
        p263 = landmarks[KinematicExtractor.FACE_LEFT]
        p152 = landmarks[KinematicExtractor.CHIN]

        # Calculate horizontal distance ratio from Nose Tip to Left/Right Eye Anchors
        dist_right = math.sqrt((p1.get('x', 0) - p33.get('x', 0))**2 + (p1.get('y', 0) - p33.get('y', 0))**2)
        dist_left = math.sqrt((p1.get('x', 0) - p263.get('x', 0))**2 + (p1.get('y', 0) - p263.get('y', 0))**2)

        total_span = dist_right + dist_left
        if total_span < 1e-6:
            return 0.0, 0.0

        # Symmetry asymmetry ratio translates to Yaw angle in degrees (-90 to +90)
        asymmetry = (dist_right - dist_left) / total_span
        yaw_deg = asymmetry * 90.0

        # Vertical pitch ratio (Nose Tip to Eye Line vs Nose Tip to Chin)
        eye_mid_y = (p33.get('y', 0) + p263.get('y', 0)) / 2.0
        pitch_span = math.abs(p152.get('y', 0) - eye_mid_y)
        if pitch_span < 1e-6:
            pitch_deg = 0.0
        else:
            nose_pos = (p1.get('y', 0) - eye_mid_y) / pitch_span
            pitch_deg = (nose_pos - 0.4) * 60.0

        return yaw_deg, pitch_deg

    def extract_vector(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """
        Extracts 8-dimensional normalized kinematic vector [f1, f2, f3, f4, f5, f6, f7, f8]
        from a list of MediaPipe landmark dicts [{'x': float, 'y': float, 'z': float}, ...].
        Includes Adaptive Side-View (Profile) Pose Compensation.
        """
        if not landmarks or len(landmarks) < 315:
            return [0.0] * 8

        # Get landmark points safely
        p1 = landmarks[self.NOSE_TIP]
        p33 = landmarks[self.FACE_RIGHT]
        p263 = landmarks[self.FACE_LEFT]
        p61 = landmarks[self.LIP_CORNER_RIGHT]
        p291 = landmarks[self.LIP_CORNER_LEFT]
        p0 = landmarks[self.UPPER_LIP_CENTER]
        p37 = landmarks[self.UPPER_LIP_MID_RIGHT]
        p267 = landmarks[self.UPPER_LIP_MID_LEFT]
        p13 = landmarks[self.UPPER_LIP_INNER]
        p14 = landmarks[self.LOWER_LIP_INNER]
        p17 = landmarks[self.LOWER_LIP_CENTER]
        p84 = landmarks[self.LOWER_LIP_MID_RIGHT]
        p314 = landmarks[self.LOWER_LIP_MID_LEFT]

        # 1. Estimate Head Pose (Yaw) to detect Frontal vs Side Profile
        yaw_deg, _ = self.estimate_head_pose(landmarks)
        is_side_profile = abs(yaw_deg) > 22.0

        # 2. Adaptive Normalization Base Calculation
        if not is_side_profile:
            # Frontal View: Dual-Eye Width Base
            face_width = self._euclidean_distance_2d(p33, p263)
            mouth_width = self._euclidean_distance_2d(p61, p291)
        else:
            # Profile / Side-View Compensation: Use Nose-to-Visible Anchor Base
            if yaw_deg > 0: # Turned Right -> Left side visible
                visible_span = self._euclidean_distance_2d(p1, p263)
            else: # Turned Left -> Right side visible
                visible_span = self._euclidean_distance_2d(p1, p33)
            
            face_width = visible_span * 2.1
            # Adjust mouth width for perspective compression
            raw_mouth = self._euclidean_distance_2d(p61, p291)
            perspective_factor = max(0.4, math.cos(math.radians(yaw_deg)))
            mouth_width = raw_mouth / perspective_factor

        if face_width < 1e-6:
            face_width = 1.0
        if mouth_width < 1e-6:
            mouth_width = 1.0

        # Feature 1: Vertical Aperture Ratio = ||P13 - P14|| / Face_Width
        aperture_dist = self._euclidean_distance_2d(p13, p14)
        f1 = aperture_dist / face_width

        # Feature 2: Horizontal Width Ratio = Mouth_Width / Face_Width
        f2 = mouth_width / face_width

        # Feature 3: Upper Lip Curvature
        upper_mid_y_avg = (p37.get('y', 0.0) + p267.get('y', 0.0)) / 2.0
        f3 = (p0.get('y', 0.0) - upper_mid_y_avg) / mouth_width

        # Feature 4: Lower Lip Curvature
        lower_mid_y_avg = (p84.get('y', 0.0) + p314.get('y', 0.0)) / 2.0
        f4 = (p17.get('y', 0.0) - lower_mid_y_avg) / mouth_width

        # Feature 5: Lip Corner Height Variance
        f5 = (p61.get('y', 0.0) - p291.get('y', 0.0)) / mouth_width

        # Feature 6: Aperture Velocity Delta = f1_t - f1_{t-1}
        f6 = (f1 - self.prev_f1) if self.prev_f1 is not None else 0.0

        # Feature 7: Width Velocity Delta = f2_t - f2_{t-1}
        f7 = (f2 - self.prev_f2) if self.prev_f2 is not None else 0.0

        # Feature 8: Kinematic Acceleration = f6_t - f6_{t-1}
        f8 = (f6 - self.prev_f6) if self.prev_f6 is not None else 0.0

        # Update historical state for deltas
        self.prev_f1 = f1
        self.prev_f2 = f2
        self.prev_f6 = f6

        return [
            round(f1, 5),
            round(f2, 5),
            round(f3, 5),
            round(f4, 5),
            round(f5, 5),
            round(f6, 5),
            round(f7, 5),
            round(f8, 5)
        ]
