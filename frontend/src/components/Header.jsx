import React from 'react';
import { Activity, ShieldAlert, Globe, Cpu, Volume2, Zap, Radio } from 'lucide-react';

export default function Header({
  isSystemOnline,
  language,
  setLanguage,
  voiceMode,
  setVoiceMode,
  wsStatus,
  onTriggerSOS
}) {
  return (
    <header className="w-full bg-slate-900/80 backdrop-blur-md border-b border-slate-800/80 px-6 py-3.5 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between flex-wrap gap-3">
        
        {/* Brand & Logo */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Volume2 className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                DeepVani
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-full uppercase tracking-wider">
                Silent Speech AI
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Neural Sub-vocalization & Speech Synthesis Platform
            </p>
          </div>
        </div>

        {/* Center/Right Status & Controls */}
        <div className="flex items-center space-x-3 flex-wrap gap-2">
          
          {/* WebSocket Duplex Indicator */}
          <div className="flex items-center space-x-1.5 bg-slate-800/60 px-2.5 py-1 rounded-lg border border-slate-700/50 text-[11px]">
            <Radio className={`w-3 h-3 ${wsStatus === 'connected' ? 'text-emerald-400 animate-pulse' : 'text-slate-500'}`} />
            <span className="text-slate-300 font-mono">
              {wsStatus === 'connected' ? 'WS Duplex' : 'REST Mode'}
            </span>
          </div>

          {/* Voice Mode Toggle (Instant <10ms vs Neural) */}
          <div className="flex items-center bg-slate-800/80 rounded-lg p-0.5 border border-slate-700/60 text-xs">
            <button
              onClick={() => setVoiceMode('instant')}
              className={`flex items-center space-x-1 px-2.5 py-1 font-semibold rounded-md transition-all ${
                voiceMode === 'instant'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Zero-latency browser speech synthesis (<10ms)"
            >
              <Zap className="w-3 h-3 text-amber-400" />
              <span>Instant (&lt;10ms)</span>
            </button>

            <button
              onClick={() => setVoiceMode('neural')}
              className={`flex items-center space-x-1 px-2.5 py-1 font-semibold rounded-md transition-all ${
                voiceMode === 'neural'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="High-fidelity Azure Neural Studio Voice (~1.3s)"
            >
              <Volume2 className="w-3 h-3" />
              <span>Neural Studio</span>
            </button>
          </div>

          {/* Language Selector Toggle */}
          <div className="flex items-center bg-slate-800/80 rounded-lg p-0.5 border border-slate-700/60">
            <button
              onClick={() => setLanguage('hi')}
              className={`flex items-center space-x-1 px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
                language === 'hi'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span>🇮🇳 Hindi</span>
            </button>
            <button
              onClick={() => setLanguage('en')}
              className={`flex items-center space-x-1 px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
                language === 'en'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span>🌐 English</span>
            </button>
          </div>

          {/* S.O.S Emergency Button */}
          <button
            onClick={onTriggerSOS}
            className="flex items-center space-x-1.5 bg-rose-600/20 hover:bg-rose-600 text-rose-400 hover:text-white px-3 py-1 rounded-lg border border-rose-500/40 text-xs font-semibold transition-all shadow-md shadow-rose-950/40"
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>S.O.S</span>
          </button>

        </div>

      </div>
    </header>
  );
}
