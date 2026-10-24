# DeepVani ML Subsystem Documentation
## Developer Guide for Computer Vision & Backend Teammates

This directory contains the machine learning core of **DeepVani**, implementing real-time 3D kinematic feature vectorization, temporal sliding window classification, LLM context disambiguation, and neural speech synthesis.

---

## 🛠️ Module Architecture

```
backend/
├── ml/
│   ├── kinematics.py          # Extracts 8-D normalized lip metric vectors from MediaPipe landmarks
│   ├── dataset_generator.py   # Produces synthetic temporal tensor datasets (30x8) for 15 core phrases
│   ├── classifier.py          # KinematicBuffer queue & SilentSpeechClassifier ML model
│   └── README.md              # This guide for CV & Backend developers
├── services/
│   ├── agent_service.py       # Groq Llama-3-8B context disambiguator & sentence builder
│   └── tts_service.py         # Edge-TTS neural speech synthesizer wrapper
├── models/
│   └── silent_speech_model.pkl# Serialized binary model weights (generated via train_model.py)
├── train_model.py             # CLI script to generate dataset, train model & save weights
└── test_ml_pipeline.py        # Integration test verifying full pipeline execution
```

---

## 🔌 Integration Guide for Teammates

### 1. For the Computer Vision (CV) Teammate
When capturing frames via OpenCV / MediaPipe Face Mesh:
1. Extract the 468 3D landmark points array: `landmarks = [{'x': float, 'y': float, 'z': float}, ...]`.
2. Pass the array to `KinematicExtractor.extract_vector(landmarks)`.
3. Push the resulting 8-dimensional list into `KinematicBuffer.push(vector)`.

```python
from backend.ml.kinematics import KinematicExtractor
from backend.ml.classifier import KinematicBuffer

extractor = KinematicExtractor()
buffer = KinematicBuffer(buffer_size=30, feature_dim=8)

# In your frame processing loop:
vector = extractor.extract_vector(mediapipe_landmarks)
is_full = buffer.push(vector)

if is_full:
    tensor_matrix = buffer.get_tensor() # Shape: (30, 8)
```

### 2. For the Backend / FastAPI Teammate
In your REST API endpoint `/api/v1/decode-silent-speech`:
1. Receive incoming frame vectors or 30-frame tensor matrix.
2. Call `classifier.predict(tensor_matrix)`.
3. If `result['passed_gating']` is True, pass `result['raw_candidate']` to `agent_service.polish_sentence()`.
4. Synthesize speech via `tts_service.generate_speech(sentence, lang)`.

```python
from backend.ml.classifier import SilentSpeechClassifier
from backend.services.agent_service import AgentDisambiguatorService
from backend.services.tts_service import TTSService

classifier = SilentSpeechClassifier()
classifier.load_model("backend/models/silent_speech_model.pkl")

agent_service = AgentDisambiguatorService()
tts_service = TTSService()

# Prediction flow:
ml_result = classifier.predict(tensor_matrix)

if ml_result["passed_gating"]:
    polished_sentence = agent_service.polish_sentence(
        raw_candidate=ml_result["raw_candidate"],
        language=ml_result["language"]
    )
    audio_url = tts_service.generate_speech(polished_sentence, ml_result["language"])
```

---

## ⚡ Commands to Train & Test

### Train Model & Save Weights:
```bash
python backend/train_model.py
```

### Run End-to-End Pipeline Test:
```bash
python backend/test_ml_pipeline.py
```
