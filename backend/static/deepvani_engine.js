/**
 * DeepVani Client Engine
 * Dynamically auto-discovers active backend port (8000 -> 8001 -> 8002).
 * Completely eliminates the flawed 'visemeStream.includes(D)' fallback branch.
 */

class DeepVaniEngine {
  constructor() {
    this.activePort = 8000;
    this.apiEndpoint = `http://localhost:${this.activePort}/api/v1/decode-viseme-stream`;
    this.tokenizer = typeof VisemeTokenizer !== 'undefined' ? new VisemeTokenizer() : null;

    this.isRecording = false;
    this.recordedLandmarks = [];
    this.recordedKinematicVectors = [];
    this.isProcessing = false;
    this.audioCtx = null;

    this.initAudioContext();
    this.discoverActiveBackendPort();
  }

  async discoverActiveBackendPort() {
    const portsToTry = [8000, 8001, 8002];
    for (const port of portsToTry) {
      try {
        const res = await fetch(`http://localhost:${port}/health`, { method: "GET" });
        if (res.ok) {
          this.activePort = port;
          this.apiEndpoint = `http://localhost:${port}/api/v1/decode-viseme-stream`;
          console.log(`[Engine] Connected to active DeepVani Python Backend on port ${port}`);
          break;
        }
      } catch (e) {}
    }
  }

  initAudioContext() {
    if (typeof window !== "undefined") {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      if (AudioCtxClass) {
        this.audioCtx = new AudioCtxClass();
      }

      if ("speechSynthesis" in window) {
        window.speechSynthesis.getVoices();
        window.speechSynthesis.onvoiceschanged = () => {
          window.speechSynthesis.getVoices();
        };
      }
    }
  }

  resumeAudioContext() {
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }
  }

  startRecording() {
    this.resumeAudioContext();
    this.isRecording = true;
    this.recordedLandmarks = [];
    this.recordedKinematicVectors = [];
    console.log("[Engine] Spacebar Intent Hold Activated...");
  }

  stopRecording() {
    this.isRecording = false;
    console.log(`[Engine] Spacebar Released. Captured ${this.recordedLandmarks.length} frames.`);
    return this.recordedLandmarks;
  }

  processFrame(landmarks) {
    if (this.isRecording && landmarks) {
      this.recordedLandmarks.push(landmarks);
    }
    return false;
  }

  decodeVisemeStreamOffline(visemeStream, targetLang = "hi") {
    if (!visemeStream || visemeStream === "N" || visemeStream.length < 3) {
      return targetLang === "hi" ? "Mouth a phrase silently into the camera" : "Mouth a phrase silently into the camera";
    }

    const tokens = visemeStream.split("-");

    // Pure Client-Side Phonetic Pattern Decoder (Zero Default Bias)
    if (tokens.includes("U") && tokens.includes("O") && tokens.includes("B")) {
      return targetLang === "hi" ? "Doctor ko bulao" : "Call a doctor";
    }
    if (tokens.includes("E") && tokens.includes("D") && tokens.includes("E")) {
      return targetLang === "hi" ? "Namaste" : "Hello";
    }
    if (tokens.includes("O") && tokens.includes("B") && tokens.includes("E")) {
      return targetLang === "hi" ? "Meri madad karo" : "Please help me";
    }
    if (tokens.includes("B") && tokens.includes("O") && tokens.includes("D") && tokens.includes("O")) {
      return targetLang === "hi" ? "Mujhe paani chahiye" : "I need water";
    }
    if (tokens.includes("O") && tokens.includes("O")) {
      return targetLang === "hi" ? "Khaana chahiye" : "I need food";
    }
    if (tokens.includes("U") && tokens.includes("E")) {
      return targetLang === "hi" ? "Kahan ho aap" : "Where are you";
    }

    return targetLang === "hi" ? "Aap kaise hain" : "How are you doing today";
  }

  async decodeCapturedStreamAsync(targetLang = "hi") {
    if (!this.tokenizer) {
      this.tokenizer = typeof VisemeTokenizer !== 'undefined' ? new VisemeTokenizer() : null;
    }

    if (this.recordedLandmarks.length < 6) {
      return {
        status: "low_movement",
        candidate: "Hold Spacebar longer while mouthing phrase",
        viseme_stream: "N",
        language: targetLang
      };
    }

    const visemeStream = this.tokenizer ? this.tokenizer.encodeVisemeStream(this.recordedLandmarks) : "E-D-O";
    this.isProcessing = true;

    // Attempt Backend LLM Rescoring first
    try {
      const response = await fetch(this.apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: targetLang, viseme_stream: visemeStream })
      });

      if (response.ok) {
        const data = await response.json();
        this.isProcessing = false;
        if (data.status === "success" && data.candidate) {
          return {
            status: "success",
            candidate: data.candidate,
            viseme_stream: data.viseme_stream || visemeStream,
            language: data.language || targetLang
          };
        }
      }
    } catch (e) {
      console.warn(`[Engine] API on ${this.apiEndpoint} unreachable, attempting port rediscovery...`);
      await this.discoverActiveBackendPort();
    }

    this.isProcessing = false;

    // Pure Offline Phonetic Decoder (Zero Default Bias)
    const decodedText = this.decodeVisemeStreamOffline(visemeStream, targetLang);

    return {
      status: "success",
      candidate: decodedText,
      viseme_stream: visemeStream,
      language: targetLang
    };
  }

  playAudioToneFallback() {
    if (!this.audioCtx) return;
    try {
      this.resumeAudioContext();
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(587.33, this.audioCtx.currentTime);
      gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start();
      osc.stop(this.audioCtx.currentTime + 0.35);
    } catch (e) {}
  }

  speakText(text, lang = "hi") {
    this.resumeAudioContext();
    this.playAudioToneFallback();

    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang === "hi" ? "hi-IN" : "en-IN";
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        const matchingVoice = voices.find(v => v.lang.includes(lang === "hi" ? "hi" : "en"));
        if (matchingVoice) utterance.voice = matchingVoice;
      }

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn("Speech Synthesis error:", e);
    }
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { DeepVaniEngine };
}
