import React, { useState, useEffect, useRef } from 'react';
import { X, Mic, CheckCircle, Sparkles, Loader2, AlertCircle } from 'lucide-react';
import { quickTrainPhrase } from '../services/api';

export default function QuickTrainModal({ isOpen, onClose, onTrainSuccess, latestVector }) {
  const [phraseLabel, setPhraseLabel] = useState('');
  const [language, setLanguage] = useState('hi');
  const [step, setStep] = useState(1); // 1: Input, 2: Record 3 Samples, 3: Success
  const [samplesRecorded, setSamplesRecorded] = useState(0);
  const [isRecordingSample, setIsRecordingSample] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [recordedTensors, setRecordedTensors] = useState([]);

  const currentSampleBuffer = useRef([]);

  // Capture live incoming vectors while sample recording is active
  useEffect(() => {
    if (isRecordingSample && latestVector && latestVector.length === 8) {
      currentSampleBuffer.current.push(latestVector);
      const progress = Math.min(100, Math.round((currentSampleBuffer.current.length / 30) * 100));
      setRecordProgress(progress);

      if (currentSampleBuffer.current.length >= 30) {
        // Sample completed!
        setIsRecordingSample(false);
        const completeTensor = currentSampleBuffer.current.slice(0, 30);
        currentSampleBuffer.current = [];

        const updatedTensors = [...recordedTensors, completeTensor];
        setRecordedTensors(updatedTensors);
        const count = samplesRecorded + 1;
        setSamplesRecorded(count);
        setRecordProgress(0);

        if (count >= 3) {
          setStep(3);
        }
      }
    }
  }, [latestVector, isRecordingSample, recordedTensors, samplesRecorded]);

  if (!isOpen) return null;

  const handleStartRecording = () => {
    setErrorMessage('');
    currentSampleBuffer.current = [];
    setRecordProgress(0);
    setIsRecordingSample(true);
  };

  const handleFinalSubmit = async () => {
    if (!phraseLabel.trim() || recordedTensors.length === 0) return;
    setIsSubmitting(true);
    setErrorMessage('');

    try {
      const res = await quickTrainPhrase(phraseLabel.trim(), recordedTensors, language);
      setIsSubmitting(false);

      if (res.status === 'success') {
        onTrainSuccess(phraseLabel.trim());
        onClose();
        // Reset modal state
        setPhraseLabel('');
        setStep(1);
        setSamplesRecorded(0);
        setRecordedTensors([]);
      } else {
        setErrorMessage(res.message || 'Failed to train custom phrase.');
      }
    } catch (e) {
      setIsSubmitting(false);
      setErrorMessage(e.message || 'Network error during model training.');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl relative overflow-hidden">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg bg-slate-800/60"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center space-x-2.5 mb-5">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center border border-cyan-500/40 text-cyan-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">Zero-Shot Phrase Trainer</h3>
            <p className="text-xs text-slate-400">Train DeepVani AI on genuine silent lip kinematics live</p>
          </div>
        </div>

        {errorMessage && (
          <div className="mb-4 p-3 bg-rose-500/20 border border-rose-500/40 rounded-xl text-rose-300 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Step 1: Input Custom Phrase */}
        {step === 1 && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Custom Phrase String
              </label>
              <input
                type="text"
                value={phraseLabel}
                onChange={(e) => setPhraseLabel(e.target.value)}
                placeholder="e.g., Calling for water"
                maxLength={80}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Target Language
              </label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="hi">Hindi (hi-IN)</option>
                <option value="en">English (en-IN)</option>
              </select>
            </div>

            <button
              disabled={!phraseLabel.trim()}
              onClick={() => setStep(2)}
              className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-semibold py-2.5 rounded-xl text-xs transition-all shadow-lg shadow-cyan-600/30"
            >
              Next: Record 3 Live Gestures
            </button>
          </div>
        )}

        {/* Step 2: Record 3 Real Live Gestures */}
        {step === 2 && (
          <div className="text-center py-4 space-y-4">
            <p className="text-sm font-semibold text-slate-200">
              Mouth silently: <span className="text-cyan-400">"{phraseLabel}"</span>
            </p>
            <p className="text-xs text-slate-400">Sample {samplesRecorded + 1} of 3</p>

            <div className="flex flex-col items-center justify-center my-4">
              <button
                disabled={isRecordingSample}
                onClick={handleStartRecording}
                className={`w-20 h-20 rounded-full border-2 flex items-center justify-center transition-all ${
                  isRecordingSample
                    ? 'bg-rose-500/20 border-rose-400 text-rose-400 animate-pulse'
                    : 'bg-cyan-500/20 border-cyan-400 text-cyan-400 hover:bg-cyan-500/30 glow-cyan'
                }`}
              >
                <Mic className="w-8 h-8" />
              </button>

              {isRecordingSample && (
                <div className="w-48 mt-3 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-cyan-400 h-full transition-all duration-75"
                    style={{ width: `${recordProgress}%` }}
                  />
                </div>
              )}
            </div>

            <p className="text-[11px] text-slate-500">
              {isRecordingSample
                ? 'Tracking mouth movement... keep mouthing the phrase!'
                : 'Click the button and mouth the phrase clearly for 1 second.'}
            </p>
          </div>
        )}

        {/* Step 3: Success Confirmation & Save */}
        {step === 3 && (
          <div className="text-center py-4 space-y-4">
            <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto animate-bounce" />
            <h4 className="text-base font-bold text-slate-100">3 Real Kinematic Samples Captured!</h4>
            <p className="text-xs text-slate-400">
              Ready to fit Random Forest weights on: <span className="text-cyan-400 font-semibold">"{phraseLabel}"</span>
            </p>

            <button
              disabled={isSubmitting}
              onClick={handleFinalSubmit}
              className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-2.5 rounded-xl text-xs transition-all flex items-center justify-center space-x-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Fitting Weights in Background...</span>
                </>
              ) : (
                <span>Register Phrase to AI Core</span>
              )}
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
