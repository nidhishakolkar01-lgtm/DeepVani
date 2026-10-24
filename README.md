<div align="center">

# 🗣️ DeepVani
### Real-Time Silent Speech Decoder & Neural Voice Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-FF6F00.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DeepVani** is an assistive AI platform engineered to decode silent speech and lip kinematics into fluent, audible language in real time. By fusing client-side 3D computer vision, kinematic feature vectorization, temporal sliding-window machine learning, LLM semantic disambiguation, and neural text-to-speech synthesis, DeepVani restores independent communication for non-vocal individuals, aphonia patients, and silent operation environments.

[Features](#-key-features) • [System Architecture](#-system-architecture) • [How It Works](#-how-it-works) • [Tech Stack](#-technology-stack) • [Installation & Setup](#-installation--setup) • [API Reference](#-api-reference) • [Performance](#-performance--latency-benchmarks)

---

</div>

## 📌 Motivation & Vision

Over 70 million individuals globally live with profound vocal impairments resulting from conditions such as ALS, stroke, tracheostomies, laryngeal trauma, or vocal cord paralysis. Existing assistive speech technologies frequently rely on physical touch screens, slow eye-gaze typing keyboards, or invasive, uncomfortable electromyography (EMG) sensors.

**DeepVani** approaches voice restoration through non-invasive optical sensing and silent speech recognition. Using any commodity consumer webcam (30/60 FPS), DeepVani tracks the subtle kinematic movements of the lips and mouth, predicts unspoken phrases with high temporal resolution, semantically polishes the intended message with an LLM, and speaks it aloud with low-latency neural TTS.

---

## 🌟 Key Features

- **Passive Optical Sensing**: Operates entirely with standard webcams—no specialized biometric sensors, electrodes, or wearable hardware required.
- **3D Kinematic Lip Vectorization**: Computes normalized 8-dimensional geometric tensors (`horizontal_spread`, `vertical_opening`, `lip_thickness`, `outer_perimeter`, `aspect_ratio`, etc.) invariant to head rotation and camera distance.
- **Sequence-Aware Temporal Classifier**: Temporal sliding-window engine aggregating continuous 30-frame batches (30x8 tensors) with custom confidence-gating mechanisms to suppress accidental mouth twitches and breathing artifacts.
- **LLM Semantic Disambiguation**: Integrated Groq / LLaMA-3 agent disambiguates visually homophenous words, corrects syntactical gaps, and constructs natural multi-lingual sentences (Hindi & English).
- **Neural Voice Synthesis**: Sub-second speech generation using Microsoft Edge neural voice models with customizable pitch, speed, and regional language accents.
- **Dual Runtime Interface**:
  - **Modern React + Vite Dashboard**: High-fidelity UI with real-time MediaPipe canvas overlays, real-time confidence telemetry, quick-training calibration modals, and phrase history.
  - **Standalone Desktop / Electron Client**: Self-contained web or desktop executable bundled with lightweight JavaScript viseme tokenizers.

---

## 🏗️ System Architecture

```
                          [ 📹 Camera Feed ]
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  Client-Side Vision Layer (MediaPipe Face Mesh) │
          │   - 468 Facial 3D Landmarks                     │
          │   - Real-Time Mesh Overlay (HTML5 Canvas)       │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │   Kinematic Extraction & Feature Vectorization  │
          │   - 8-D Lip Metric Normalization (30-frame FIFO)│
          │   - Tensor Matrix Construction: Shape (30, 8)   │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼  (REST / WebSocket API)
          ┌─────────────────────────────────────────────────┐
          │      FastAPI High-Performance Backend Core      │
          ├─────────────────────────────────────────────────┤
          │  1. Temporal ML Classifier                      │
          │     - Sequence-aware Random Forest / Multi-Class│
          │     - Confidence Gating & Threshold Filter      │
          │                                                 │
          │  2. LLM Disambiguator (Groq / LLaMA-3-8B)       │
          │     - Homophene Disambiguation                  │
          │     - Sentence Polishing & Contextual Naturalness│
          │                                                 │
          │  3. Neural TTS Synthesizer (Edge-TTS)           │
          │     - Streamlined MP3 Audio Streaming           │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
                    [ 🔊 Real-Time Audio Output ]
```

---

## 🔍 How It Works

### 1. 3D Lip Kinematics Extraction
The client ingests continuous frames from the webcam and tracks 468 3D facial landmarks. Key anatomical lip indices (outer contour, inner contour, vermilion borders, oral commissures) are normalized against interpupillary distance and face bounding dimensions to yield an 8-dimensional metric vector per frame:
1. `horizontal_aperture`: Normalized oral commissure span.
2. `vertical_aperture`: Inner upper to lower vermilion distance.
3. `outer_lip_height`: Total vertical height including vermilion borders.
4. `aspect_ratio`: Ratio of vertical to horizontal opening.
5. `upper_lip_curvature`: Curvature offset relative to philtrum.
6. `lower_lip_curvature`: Curvature offset relative to mentolabial sulcus.
7. `lip_perimeter_ratio`: Normalized circumference metric.
8. `area_approximation`: Inner lip elliptical area.

### 2. Sliding Temporal Window
Frame vectors are streamed into a 30-frame First-In-First-Out (FIFO) tensor buffer. When the buffer fills, a normalized `(30, 8)` matrix representing ~1 second of articulatory movement is evaluated by the classification engine.

### 3. Machine Learning & Gating
The classifier processes the spatio-temporal matrix against a pre-trained phrase library. To eliminate false triggers from passive facial twitches, a strict **confidence-gating layer** evaluates class probability distributions, verifying that peak confidence exceeds defined thresholds before dispatching candidates.

### 4. Semantic LLM Disambiguation
Silent speech frequently suffers from *homopheny* (words that look identical on the lips, e.g., "pat", "bat", "mat"). The raw candidate is routed to the `AgentDisambiguatorService`, where a prompt-tuned LLaMA-3 model validates grammatical flow, resolves linguistic context, and renders a coherent statement.

### 5. Multi-Lingual Speech Synthesis
The disambiguated statement is converted into natural human speech via asynchronous neural TTS streaming, returning an audio stream that plays through the user's audio output.

---

## 💻 Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend Framework** | React 18, Vite 5, Tailwind CSS, Lucide Icons |
| **Desktop Client** | Electron, HTML5 Canvas, Vanilla JavaScript |
| **Computer Vision** | Google MediaPipe Face Mesh, OpenCV |
| **Backend Framework** | FastAPI (ASGI), Uvicorn, Pydantic v2 |
| **Machine Learning** | NumPy, Scikit-Learn, LightGBM, Pickle |
| **LLM Inference** | Groq Cloud API, LLaMA-3 8B Instant |
| **Voice Synthesis** | Microsoft Edge-TTS (Multi-lingual neural voices) |
| **Testing & Quality** | Pytest, Pyrefly, ESLint |

---

## 📁 Repository Structure

```
DeepVani/
├── backend/
│   ├── config.py                 # Pydantic BaseSettings & environment loader
│   ├── main.py                   # FastAPI REST API & WebSocket handlers
│   ├── server.py                 # Standalone backend server runner
│   ├── check_app.py              # Health check & smoke verification utility
│   ├── live_demo_client.py       # Terminal client for real-time inference testing
│   ├── test_ml_pipeline.py       # Full end-to-end ML pipeline test script
│   ├── train_model.py            # Dataset compilation & model training script
│   ├── data/
│   │   └── real_lip_dataset.json # Calibrated human kinematic landmark recordings
│   ├── ml/
│   │   ├── kinematics.py         # 3D MediaPipe landmark vectorizer (8-D metrics)
│   │   ├── classifier.py         # SilentSpeechClassifier & KinematicBuffer
│   │   ├── nn_classifier.py      # Neural network classification module
│   │   ├── dataset_generator.py  # Synthetic and real temporal dataset synthesizers
│   │   ├── augmenter.py          # Kinematic tensor noise & jitter augmentation
│   │   ├── camera_simulator.py   # Synthetic video stream test generator
│   │   └── ml_engine.py          # Central machine learning coordinator
│   ├── models/
│   │   ├── silent_speech_model.pkl  # Trained classifier binary weights
│   │   └── silent_speech_model.json # Model metadata and class label index
│   ├── services/
│   │   ├── agent_service.py      # Groq LLaMA-3 contextual disambiguation service
│   │   ├── tts_service.py        # Asynchronous Edge-TTS neural speech generator
│   │   └── extractor.py          # Landmark extraction & geometry utilities
│   ├── static/                   # Static audio assets and standalone web assets
│   └── tests/                    # Unit tests for agent, classifier & kinematics
├── frontend/                     # Modern React + Vite application
│   ├── src/
│   │   ├── App.jsx               # Primary application state coordinator
│   │   ├── main.jsx              # DOM entrypoint
│   │   ├── index.css             # Tailwind base & custom design tokens
│   │   ├── components/
│   │   │   ├── Header.jsx        # Telemetry, status badges & language switcher
│   │   │   ├── MeshCanvas.jsx    # Real-time WebGL/Canvas MediaPipe overlay
│   │   │   ├── QuickTrainModal.jsx # In-app custom phrase calibration modal
│   │   │   ├── TranscriptView.jsx # Live decoded sentences & history cards
│   │   │   └── VoiceControls.jsx # TTS voice customization & audio playback
│   │   └── services/
│   │       └── api.js            # Axios client with resilient error handling
│   ├── package.json              # Frontend npm dependencies
│   ├── tailwind.config.js        # Design system styles & color palette
│   └── vite.config.js            # Vite build configuration
├── index.html                    # Standalone lightweight web client
├── deepvani_engine.js            # Browser MediaPipe camera & landmark runner
├── viseme_tokenizer.js           # Lightweight client-side viseme classifier
├── main.js                       # Electron window process manager
├── requirements.txt              # Backend Python dependencies
├── package.json                  # Root desktop project configuration
└── run_deepvani.bat              # One-click Windows development launcher
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python**: Version `3.10` or higher
- **Node.js**: Version `18.x` or higher
- **Webcam**: Standard 720p/1080p camera

### 1. Clone the Repository
```bash
git clone https://github.com/nidhishakolkar01-lgtm/DeepVani.git
cd DeepVani
```

### 2. Configure Backend
```bash
# Create and activate a Python virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

Configure your `.env` file:
```env
DEEPVANI_API_KEY=deepvani-local-dev-key-2026
GROQ_API_KEY=your_groq_api_key_here
```
*(Note: If `GROQ_API_KEY` is not provided, the application automatically falls back to rule-based phrase parsing without disruption).*

### 3. Launch Backend Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The FastAPI interactive documentation will be accessible at `http://localhost:8000/docs`.

### 4. Configure & Launch Frontend
In a separate terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## ⚡ Quick Run Scripts (Windows)

For single-command convenience on Windows systems:
- **`run_deepvani.bat`**: Initializes the backend API and launches the local web dashboard.
- **`run_electron_app.bat`**: Launches DeepVani as a native desktop application via Electron.
- **`run_localhost.bat`**: Serves the lightweight standalone HTML5/JS client on port 3000.

---

## 📡 API Reference

### Health & Telemetry
```http
GET /api/v1/health
```
**Response:**
```json
{
  "status": "healthy",
  "project": "DeepVani - Silent Speech Decoder & Neural Voice Platform",
  "model_loaded": true,
  "total_real_samples": 45,
  "timestamp": 1729810800
}
```

### Decode Silent Speech Tensor
```http
POST /api/v1/decode-silent-speech
Content-Type: application/json
```
**Payload:**
```json
{
  "language": "hi",
  "frames_vector": [
    [0.45, 0.12, 0.28, 0.26, 0.05, -0.02, 1.15, 0.08],
    "... 29 more 8-dimensional frame vectors ..."
  ]
}
```
**Response:**
```json
{
  "status": "success",
  "phrase_id": 3,
  "candidate": "मुझे पानी चाहिए",
  "polished_sentence": "मुझे पीने के लिए पानी चाहिए।",
  "language": "hi",
  "confidence": 0.94,
  "audio_url": "/static/audio/speech_961a62be6b914812.mp3",
  "latency_ms": 182.4
}
```

### Record Calibration Sample
```http
POST /api/v1/record-real-lip-sample
Content-Type: application/json
```
Saves custom user-recorded 3D landmark sequences to train individualized vocabulary.

---

## ⏱️ Performance & Latency Benchmarks

Tested on standard consumer hardware (Intel Core i7-12700H, 16GB RAM, integrated camera):

| Pipeline Stage | Processing Unit | Latency |
|---|---|---|
| **Facial Landmark Detection** | Client (MediaPipe FaceMesh) | ~12 - 16 ms |
| **Kinematic Feature Extraction** | Client JavaScript / NumPy | ~2 - 4 ms |
| **Temporal Tensor Inference** | Backend Scikit-Learn | ~4 - 7 ms |
| **LLM Disambiguation** | Groq Cloud LLaMA-3-8B | ~140 - 220 ms |
| **Edge-TTS Audio Synthesis** | Backend Async Streamer | ~120 - 180 ms |
| **Total End-to-End Latency** | Full Loop (Motion → Sound) | **< 450 ms** |

---

## 🧪 Testing

Execute automated unit and integration tests across the codebase:
```bash
# Run all tests
pytest backend/tests/ -v

# Run full ML pipeline integration test
python backend/test_ml_pipeline.py
```

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push -u origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <sub>Developed by <b>Nidhish Akolkar</b> • Empowering non-vocal communication through Assistive AI</sub>
</div>
