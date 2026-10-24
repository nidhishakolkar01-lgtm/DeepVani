"""
DeepVani Services - Server-side Computer Vision Landmark Extractor
Processes raw image bytes or OpenCV frame matrices and extracts 3D face mesh landmarks using MediaPipe.
"""

from typing import List, Dict, Optional

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class ServerVisionExtractor:
    """
    Server-side MediaPipe Face Mesh extractor for processing optical image frames.
    """

    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Warning: Failed to initialize MediaPipe FaceMesh on server: {e}")

    def extract_landmarks_from_bytes(self, image_bytes: bytes) -> List[Dict[str, float]]:
        """
        Converts raw JPEG/PNG bytes into OpenCV BGR frame and extracts 468 3D landmarks.
        """
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            return []

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return []

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return []

            face_landmarks = results.multi_face_landmarks[0]
            landmark_list = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in face_landmarks.landmark
            ]

            return landmark_list

        except Exception as e:
            print(f"Server vision processing error: {e}")
            return []

    def close(self):
        """Releases MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
