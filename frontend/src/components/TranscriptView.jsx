import React from 'react';
import { MessageSquareText, Sparkles, Clock, CheckCircle2, AlertCircle } from 'lucide-react';

export default function TranscriptView({ currentResult, history, isLoading }) {
  return (
    <div className="w-full bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-5 flex flex-col h-full shadow-xl">
      
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <MessageSquareText className="w-5 h-5 text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Live AI Speech Transcript
          </h2>
        </div>
        {isLoading && (
          <div className="flex items-center space-x-1.5 text-xs text-cyan-400 animate-pulse">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Groq Llama-3 Resolving...</span>
          </div>
        )}
      </div>

      {/* Main Real-Time Decoded Sentence Box */}
      <div className="my-5 p-5 rounded-xl bg-slate-950/80 border border-slate-800/80 relative overflow-hidden min-h-[120px] flex flex-col justify-between">
        
        {currentResult ? (
          <>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {currentResult.language === 'hi' ? 'Hindi Output' : 'English Output'}
                </span>
                <span className="text-xs font-mono text-emerald-400 flex items-center space-x-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>{(currentResult.confidence * 100).toFixed(1)}% Confidence</span>
                </span>
              </div>
              
              <p className="text-xl font-bold text-slate-100 tracking-wide leading-relaxed">
                "{currentResult.polished_sentence || currentResult.raw_candidate}"
              </p>
            </div>

            <div className="mt-3 pt-2 border-t border-slate-900/60 flex items-center justify-between text-xs text-slate-400">
              <span className="font-mono text-[11px]">Raw Viseme: {currentResult.raw_candidate}</span>
              <span className="font-mono text-[11px] text-cyan-400">{currentResult.latency_ms} ms</span>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center py-6 text-slate-500">
            <Sparkles className="w-8 h-8 mb-2 text-slate-700 animate-pulse" />
            <p className="text-sm font-medium">Mouth a phrase silently into the camera</p>
            <p className="text-xs text-slate-600 mt-1">DeepVani AI will decode sub-vocalizations in real time</p>
          </div>
        )}

      </div>

      {/* History Log Timeline */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
          <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center space-x-1">
            <Clock className="w-3.5 h-3.5 text-slate-500" />
            <span>Speech History Log</span>
          </span>
          <span className="font-mono text-[10px] bg-slate-800 px-2 py-0.5 rounded">{history.length} records</span>
        </div>

        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {history.length === 0 ? (
            <p className="text-xs text-slate-600 italic text-center py-4">No recent transcripts</p>
          ) : (
            history.map((item, idx) => (
              <div key={idx} className="p-3 rounded-lg bg-slate-800/40 border border-slate-800 flex items-center justify-between text-xs transition-all hover:bg-slate-800/70">
                <div>
                  <p className="font-semibold text-slate-200">"{item.polished_sentence}"</p>
                  <p className="text-[10px] text-slate-400 font-mono mt-0.5">Raw: {item.raw_candidate}</p>
                </div>
                <div className="text-right ml-3">
                  <span className="text-[10px] font-mono text-emerald-400 font-semibold">
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                  <p className="text-[10px] text-slate-400 font-mono">{item.timestamp}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

    </div>
  );
}
