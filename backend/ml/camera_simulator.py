"""
DeepVani ML Subsystem - Interactive Live Camera & Landmark Simulator
Simulates a live webcam stream emitting MediaPipe 3D face mesh landmarks at 30 FPS,
feeds vectors to KinematicBuffer, triggers ML classification, Groq LLM disambiguation,
and neural TTS audio synthesis in real-time.
"""

import os
import sys
import time
import random
import numpy as np

# Add project root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.ml.kinematics import KinematicExtractor
from backend.ml.dataset_generator import PHRASE_DICTIONARY, KinematicDatasetGenerator
from backend.ml.classifier import KinematicBuffer, SilentSpeechClassifier
from backend.services.agent_service import AgentDisambiguatorService
from backend.services.tts_service import TTSService


def run_camera_simulation(iterations: int = 3):
    print("=" * 70)
    print(" DeepVani Live Camera & Landmark Simulator")
    print(" Press Ctrl+C to exit simulation")
    print("=" * 70)

    # 1. Initialize ML Engine & Load Trained Model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "silent_speech_model.pkl")
    classifier = SilentSpeechClassifier(confidence_threshold=0.60)
    
    if not classifier.load_model(model_path):
        print("\n[Notice] Pre-trained model not found. Training quick in-memory model...")
        gen = KinematicDatasetGenerator()
        X, y, _ = gen.generate_dataset(samples_per_phrase=15)
        classifier.train(X, y)
        classifier.save_model(model_path)
        print("-> In-memory model trained and saved!")

    extractor = KinematicExtractor()
    buffer = KinematicBuffer(buffer_size=30, feature_dim=8)
    agent = AgentDisambiguatorService()
    tts = TTSService()

    print("\n[Simulator Active] Starting 30 FPS simulated camera landmark stream...")

    for sim_step in range(1, iterations + 1):
        target_phrase = random.choice(PHRASE_DICTIONARY)
        print(f"\n--- [Simulation Sequence #{sim_step}] User Mouthing: '{target_phrase['label']}' ---")

        # Simulate 30 frames of mouth movement
        extractor.reset_state()
        buffer.clear()

        for frame_idx in range(30):
            # Generate simulated landmark point variation for target phrase
            t = frame_idx / 30.0 * 2 * np.pi
            mock_aperture = 0.12 + 0.15 * np.abs(np.sin(t + target_phrase["id"]))
            mock_width = 0.42 + 0.10 * np.cos(t)

            mock_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(320)]
            # Inject simulated coordinates for inner lip landmarks (13, 14) and outer corners (61, 291)
            mock_landmarks[13] = {'x': 0.5, 'y': 0.5 - mock_aperture / 2.0, 'z': 0.0}
            mock_landmarks[14] = {'x': 0.5, 'y': 0.5 + mock_aperture / 2.0, 'z': 0.0}
            mock_landmarks[61] = {'x': 0.5 - mock_width / 2.0, 'y': 0.5, 'z': 0.0}
            mock_landmarks[291] = {'x': 0.5 + mock_width / 2.0, 'y': 0.5, 'z': 0.0}

            vector = extractor.extract_vector(mock_landmarks)
            is_full = buffer.push(vector)

            time.sleep(0.01)  # Simulates 30 FPS timing

        # Execute Pipeline on Full Buffer
        tensor_matrix = buffer.get_tensor()
        pred = classifier.predict(tensor_matrix)

        print(f" -> ML Model Classified    : '{pred['raw_candidate']}' (Confidence: {pred['confidence']*100:.1f}%)")

        if pred["passed_gating"]:
            sentence = agent.polish_sentence(pred["raw_candidate"], pred["language"])
            print(f" -> LLM Context Disambiguation: '{sentence}'")

            audio_url = tts.generate_speech(sentence, pred["language"])
            print(f" -> Audio Synthesized      : {audio_url}")
        else:
            print(" -> Low confidence gesture (Gating filter triggered).")

        time.sleep(1.0)

    print("\n" + "=" * 70)
    print(" Simulation Sequence Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_camera_simulation(iterations=3)
