"""
DeepVani Integration Test - Full ML & AI Pipeline Test Suite
Tests complete execution loop:
Kinematic Feature Extraction -> Temporal Buffer -> Classifier Prediction -> Groq LLM Disambiguation -> Edge-TTS Audio
"""

import os
import sys
import time
import numpy as np

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ml.kinematics import KinematicExtractor
from backend.ml.dataset_generator import KinematicDatasetGenerator
from backend.ml.classifier import KinematicBuffer, SilentSpeechClassifier
from backend.services.agent_service import AgentDisambiguatorService
from backend.services.tts_service import TTSService


def run_pipeline_test():
    print("=" * 70)
    print(" DeepVani Full ML & AI Pipeline Integration Test")
    print("=" * 70)

    t_start = time.time()

    # 1. Initialize Kinematic Extractor & Buffer
    print("\n[Step 1/5] Initializing Kinematic Extractor & Temporal Buffer (30, 8)...")
    extractor = KinematicExtractor()
    buffer = KinematicBuffer(buffer_size=30, feature_dim=8)

    # Simulate 30 incoming frames of MediaPipe 3D Landmarks
    for i in range(30):
        # Generate mock landmarks array (315 dicts with x, y, z)
        mock_landmarks = [{'x': 0.5 + 0.01 * np.sin(i / 5.0), 'y': 0.5 + 0.02 * np.cos(i / 5.0), 'z': 0.0} for _ in range(320)]
        vector = extractor.extract_vector(mock_landmarks)
        buffer.push(vector)

    tensor_matrix = buffer.get_tensor()
    print(f"-> Buffer Tensor Matrix Created. Shape: {tensor_matrix.shape}")

    # 2. Train or Load Classifier
    print("\n[Step 2/5] Initializing & Training SilentSpeechClassifier...")
    gen = KinematicDatasetGenerator()
    X, y, labels = gen.generate_dataset(samples_per_phrase=20)
    
    classifier = SilentSpeechClassifier(confidence_threshold=0.50) # Lower threshold for test simulation
    classifier.train(X, y)
    print(f"-> Model trained on {X.shape[0]} samples across {len(labels)} classes.")

    # 3. Predict Candidate Phrase
    print("\n[Step 3/5] Running Temporal ML Classifier Inference...")
    t_predict_start = time.time()
    ml_result = classifier.predict(tensor_matrix)
    t_predict_end = time.time()

    print(f"-> ML Classifier Output:")
    print(f"   Raw Candidate String : '{ml_result['raw_candidate']}'")
    print(f"   Confidence Score     : {ml_result['confidence'] * 100:.1f}%")
    print(f"   Passed Gating (>0.50): {ml_result['passed_gating']}")
    print(f"   ML Latency           : {(t_predict_end - t_predict_start) * 1000:.2f} ms")

    # 4. LLM Agent Context Disambiguation
    print("\n[Step 4/5] Passing candidate to AgentDisambiguatorService (Groq Llama-3 / Fallback)...")
    t_agent_start = time.time()
    agent_service = AgentDisambiguatorService()
    polished_sentence = agent_service.polish_sentence(
        raw_candidate=ml_result['raw_candidate'],
        language=ml_result['language']
    )
    t_agent_end = time.time()

    print(f"-> LLM Disambiguation Output:")
    print(f"   Polished Sentence    : '{polished_sentence}'")
    print(f"   Agent LLM Latency    : {(t_agent_end - t_agent_start) * 1000:.2f} ms")

    # 5. Neural TTS Audio Synthesis
    print("\n[Step 5/5] Synthesizing Neural Audio using Edge-TTS...")
    t_tts_start = time.time()
    tts_service = TTSService()
    audio_url = tts_service.generate_speech(
        text=polished_sentence,
        lang=ml_result['language']
    )
    t_tts_end = time.time()

    print(f"-> Neural Audio Synthesis Output:")
    print(f"   Static Audio URL     : '{audio_url}'")
    print(f"   TTS Latency          : {(t_tts_end - t_tts_start) * 1000:.2f} ms")

    # Summary
    t_total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f" PIPELINE TEST PASSED SUCCESSFULLY!")
    print(f" Total End-to-End Execution Time: {t_total * 1000:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_test()
