# DeepVani 🎙️
### Real-Time Silent Speech Decoder & Neural Voice Platform

**DeepVani** is an assistive AI platform that decodes silent speech and lip kinematics into spoken language in real time. Designed for individuals with speech impairments, aphonia, or for silent communication environments, DeepVani translates 3D facial and lip movements into natural, audible speech with low latency.

---

## 🌟 Key Features

- **3D Lip Kinematics Vectorization**: Extracts normalized 8-dimensional kinematic vectors from 468 MediaPipe FaceMesh landmarks in real time.
- **Temporal Sequence ML Classifier**: Classifies continuous temporal frames (30x8 tensor windows) to accurately recognize words and phrases without acoustic input.
- **LLM Context Disambiguator**: Integrated with Groq LLM to disambiguate phonetic ambiguities, restore missing syntax, and produce polished, context-aware sentences in Hindi and English.
- **Real-Time Neural Speech Synthesis**: Generates natural speech using Edge-TTS in milliseconds.
- **Dual Client Options**:
  - **Modern React + Vite Frontend**: Responsive dashboard with real-time landmark canvas, transcription history, confidence scores, and quick calibration.
  - **Lightweight Standalone / Electron App**: Self-contained client runnable in-browser or packaged via Electron.

---

## 🏗️ Architecture Overview

```
DeepVani/
├── backend/
│   ├── config.py             # App settings, environment variables & CORS
│   ├── main.py               # FastAPI application, REST endpoints & WebSocket routing
│   ├── data/                 # Recorded kinematic calibration datasets
│   ├── ml/
│   │   ├── kinematics.py     # 8-D normalized lip landmark extractor
│   │   ├── dataset_generator.py # Synthetic & real kinematic temporal generators
│   │   ├── classifier.py     # SilentSpeechClassifier & sliding buffer
│   │   ├── nn_classifier.py  # Neural network classification module
│   │   ├── ml_engine.py      # Core ML inference engine
│   │   └── augmenter.py      # Data augmentation for kinematics
│   ├── models/               # Serialized ML model weights
│   ├── services/
│   │   ├── agent_service.py  # Groq LLM disambiguator & sentence polisher
│   │   ├── tts_service.py    # Edge-TTS speech synthesizer
│   │   └── extractor.py      # Frame & landmark feature extraction
│   └── tests/                # Automated pytest test suites
├── frontend/                 # React 18 + Vite + Tailwind CSS dashboard
├── index.html                # Standalone browser/Electron client
├── deepvani_engine.js        # MediaPipe camera & landmark tracking engine
├── viseme_tokenizer.js       # Real-time viseme sequence tokenizer
├── main.js                   # Electron main process entrypoint
├── requirements.txt          # Python dependencies
└── package.json              # Node.js dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend / Electron)
- Webcam for real-time facial landmark capture

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/nidhishakolkar01-lgtm/DeepVani.git
cd DeepVani

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
copy .env.example .env
# Edit .env and supply your GROQ_API_KEY (optional, fallback rules available)

# Start backend server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup (React)
```bash
cd frontend
npm install
npm run dev
```

### 3. Quick Run (Windows Batch Scripts)
- Run `run_deepvani.bat` to launch both backend and frontend.
- Run `run_electron_app.bat` to start the desktop Electron experience.

---

## 🧪 Testing

Run automated tests on the ML pipeline and services:
```bash
pytest backend/tests/
```

---

## 📄 License
This project is licensed under the MIT License.
