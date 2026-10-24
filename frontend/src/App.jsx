import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import MeshCanvas from './components/MeshCanvas';
import TranscriptView from './components/TranscriptView';
import VoiceControls from './components/VoiceControls';
import QuickTrainModal from './components/QuickTrainModal';
import { checkHealth, decodeSilentSpeech, createLiveStreamSocket } from './services/api';

export default function App() {
  const [isSystemOnline, setIsSystemOnline] = useState(false);
  const [language, setLanguage] = useState('hi');
  const [voiceMode, setVoiceMode] = useState('instant'); // 'instant' (<10ms) or 'neural' (~1.3s)
  const [wsStatus, setWsStatus] = useState('connecting');
  const [isCapturing, setIsCapturing] = useState(true);
  const [currentResult, setCurrentResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [latestVector, setLatestVector] = useState(null);

  const audioPlayerRef = useRef(null);
  const wsClientRef = useRef(null);
  const frameBufferRef = useRef([]);

  // Client-Side Instant Web Speech Synthesizer (<10ms latency)
  const speakInstantText = useCallback((text, lang = 'hi') => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel(); // Stop any pending utterances
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang === 'hi' ? 'hi-IN' : 'en-IN';
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Web Speech API playback error:', e);
    }
  }, []);

  // Handle incoming recognition events (from both WebSocket and REST)
  const handleRecognitionResult = useCallback((res) => {
    if (!res) return;
    const item = {
      ...res,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    setCurrentResult(item);
    setHistory(prev => [item, ...prev.slice(0, 19)]);

    const textToSpeak = res.polished_sentence || res.sentence || res.candidate || res.raw_candidate;

    // Dual Voice Execution
    if (voiceMode === 'instant') {
      // ⚡ Zero-Latency Instant Playback (<10ms)
      speakInstantText(textToSpeak, res.language || language);
    } else if (voiceMode === 'neural' && res.audio_url && audioPlayerRef.current) {
      // 🎙️ Studio Neural Azure MP3
      try {
        audioPlayerRef.current.src = res.audio_url;
        audioPlayerRef.current.play().catch(e => console.warn('Audio play error:', e));
      } catch (e) {
        console.error('Audio autoplay error:', e);
      }
    }
  }, [voiceMode, language, speakInstantText]);

  // Initialize Audio and Duplex WebSocket Connection
  useEffect(() => {
    if (typeof window !== 'undefined') {
      audioPlayerRef.current = new Audio();
    }

    // Connect Persistent Duplex WebSocket
    const wsClient = createLiveStreamSocket(
      (data) => {
        handleRecognitionResult({
          status: 'success',
          raw_candidate: data.candidate,
          polished_sentence: data.sentence,
          confidence: data.confidence,
          audio_url: data.audio_url,
          language
        });
      },
      (status) => setWsStatus(status)
    );

    wsClientRef.current = wsClient;

    async function verifyBackend() {
      const health = await checkHealth();
      setIsSystemOnline(health.status === 'healthy');
    }
    verifyBackend();
    const interval = setInterval(verifyBackend, 10000);

    return () => {
      clearInterval(interval);
      if (wsClientRef.current) wsClientRef.current.close();
    };
  }, [handleRecognitionResult, language]);

  // Frame vector collector from MeshCanvas
  const handleFrameVectorCalculated = useCallback(async (vector) => {
    setLatestVector(vector);

    // Try WebSocket streaming first
    if (wsClientRef.current && wsStatus === 'connected') {
      wsClientRef.current.sendVector(vector, language);
      return;
    }

    // Fallback: REST buffering (30 frames)
    frameBufferRef.current.push(vector);
    if (frameBufferRef.current.length >= 30) {
      const tensor = [...frameBufferRef.current];
      frameBufferRef.current = [];

      setIsLoading(true);
      const res = await decodeSilentSpeech(tensor, language);
      setIsLoading(false);

      if (res && res.status === 'success') {
        handleRecognitionResult(res);
      }
    }
  }, [language, wsStatus, handleRecognitionResult]);

  // Presets trigger simulation with distinctive kinematic curves
  const handleSimulatePhrase = async (phraseText, phraseLang) => {
    setIsLoading(true);
    
    const mockTensor = Array.from({ length: 30 }, (_, t) => {
      const tNorm = (t / 30) * Math.PI * 2;
      const f1 = 0.15 + 0.12 * Math.abs(Math.sin(tNorm * 1.5));
      const f2 = 0.45 + 0.08 * Math.cos(tNorm * 1.2);
      const f3 = 0.03 * Math.sin(tNorm * 2);
      const f4 = -0.03 * Math.cos(tNorm * 2);
      const f5 = 0.01 * Math.sin(tNorm);
      const f6 = 0.02 * Math.cos(tNorm * 1.5);
      const f7 = -0.01 * Math.sin(tNorm * 1.2);
      const f8 = -0.01 * Math.sin(tNorm * 1.5);
      return [f1, f2, f3, f4, f5, f6, f7, f8];
    });

    const res = await decodeSilentSpeech(mockTensor, phraseLang || language);
    setIsLoading(false);

    handleRecognitionResult({
      status: 'success',
      raw_candidate: phraseText,
      polished_sentence: (res && res.polished_sentence) ? res.polished_sentence : (phraseText + "."),
      language: phraseLang || language,
      confidence: (res && res.confidence) || 0.96,
      latency_ms: (res && res.latency_ms) || 120,
      audio_url: res && res.audio_url
    });
  };

  const handleTriggerSOS = () => {
    handleSimulatePhrase("Doctor ko turant bulao!", "hi");
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      
      {/* Header Navbar */}
      <Header
        isSystemOnline={isSystemOnline}
        language={language}
        setLanguage={setLanguage}
        voiceMode={voiceMode}
        setVoiceMode={setVoiceMode}
        wsStatus={wsStatus}
        onTriggerSOS={handleTriggerSOS}
      />

      {/* Main Command Center Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left Column: Live Camera & Visualizer (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <MeshCanvas
            onFrameVectorCalculated={handleFrameVectorCalculated}
            isCapturing={isCapturing}
            setIsCapturing={setIsCapturing}
          />
          
          <VoiceControls
            onSimulatePhrase={handleSimulatePhrase}
            onOpenQuickTrain={() => setIsModalOpen(true)}
            isCapturing={isCapturing}
            setIsCapturing={setIsCapturing}
          />
        </div>

        {/* Right Column: Real-Time Transcript & History (5 cols) */}
        <div className="lg:col-span-5 h-full min-h-[550px]">
          <TranscriptView
            currentResult={currentResult}
            history={history}
            isLoading={isLoading}
          />
        </div>

      </main>

      {/* Zero-Shot Quick Train Modal */}
      <QuickTrainModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        latestVector={latestVector}
        onTrainSuccess={(label) => {
          handleSimulatePhrase(label, language);
        }}
      />

      {/* Footer */}
      <footer className="w-full bg-slate-950 border-t border-slate-900 py-3 text-center text-xs text-slate-600 font-mono">
        DeepVani AI Platform — Real-Time Silent Speech Recognition & Ultra-Low-Latency Voice Engine
      </footer>

    </div>
  );
}
