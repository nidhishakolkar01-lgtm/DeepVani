import React from 'react';
import { Play, Mic, PlusCircle, Volume2, Sparkles, Zap } from 'lucide-react';

export default function VoiceControls({ onSimulatePhrase, onOpenQuickTrain, isCapturing, setIsCapturing }) {
  const PRESET_PHRASES = [
    { label: "Mujhe paani chahiye", lang: "hi" },
    { label: "Doctor ko bulao", lang: "hi" },
    { label: "Help me", lang: "en" },
    { label: "Namaste", lang: "hi" },
    { label: "Call emergency", lang: "en" }
  ];

  return (
    <div className="w-full bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        
        {/* Presets & Simulator Triggers */}
        <div>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>Preset Silent Phrase Triggers (Demo Simulator)</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {PRESET_PHRASES.map((item, i) => (
              <button
                key={i}
                onClick={() => onSimulatePhrase(item.label, item.lang)}
                className="px-3 py-1.5 bg-slate-800/80 hover:bg-cyan-600/30 text-slate-200 hover:text-cyan-300 text-xs font-medium rounded-lg border border-slate-700/60 hover:border-cyan-500/50 transition-all flex items-center space-x-1.5 active:scale-95"
              >
                <Play className="w-3 h-3 text-cyan-400" />
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Custom Zero-Shot Train Modal Trigger */}
        <div className="flex items-center space-x-3 self-end sm:self-center">
          <button
            onClick={onOpenQuickTrain}
            className="flex items-center space-x-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-4 py-2 rounded-xl text-xs font-semibold shadow-lg shadow-cyan-500/25 transition-all active:scale-95"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Train Custom Phrase</span>
          </button>
        </div>

      </div>
    </div>
  );
}
