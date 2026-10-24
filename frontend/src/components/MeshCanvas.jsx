import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera as CameraIcon, Eye, Scan, AlertCircle } from 'lucide-react';

export default function MeshCanvas({ onFrameVectorCalculated, isCapturing, setIsCapturing }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceMeshRef = useRef(null);
  const cameraRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [fps, setFps] = useState(0);

  // Landmark Indices matching backend kinematics.py
  const FACE_RIGHT = 33;
  const FACE_LEFT = 263;
  const LIP_CORNER_RIGHT = 61;
  const LIP_CORNER_LEFT = 291;
  const UPPER_LIP_CENTER = 0;
  const UPPER_LIP_MID_RIGHT = 37;
  const UPPER_LIP_MID_LEFT = 267;
  const UPPER_LIP_INNER = 13;
  const LOWER_LIP_INNER = 14;
  const LOWER_LIP_CENTER = 17;
  const LOWER_LIP_MID_RIGHT = 84;
  const LOWER_LIP_MID_LEFT = 314;

  const OUTER_LIP_PATH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 291, 321, 314, 17, 84, 181, 91, 61];
  const INNER_LIP_PATH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 308, 402, 317, 14, 87, 178, 88, 78];

  // State refs for kinematic velocities & FPS
  const prevF1Ref = useRef(null);
  const prevF2Ref = useRef(null);
  const prevF6Ref = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());

  const euclideanDist = (p1, p2) => {
    const dx = (p1.x || 0) - (p2.x || 0);
    const dy = (p1.y || 0) - (p2.y || 0);
    return Math.sqrt(dx * dx + dy * dy);
  };

  const handleFaceMeshResults = useCallback((results) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Draw video feed
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    // Calculate real FPS
    frameCountRef.current++;
    const now = performance.now();
    if (now - lastFpsTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;
    }

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      setFaceDetected(true);
      const landmarks = results.multiFaceLandmarks[0];

      if (landmarks && landmarks.length >= 315) {
        const p33 = landmarks[FACE_RIGHT];
        const p263 = landmarks[FACE_LEFT];
        const p61 = landmarks[LIP_CORNER_RIGHT];
        const p291 = landmarks[LIP_CORNER_LEFT];
        const p0 = landmarks[UPPER_LIP_CENTER];
        const p37 = landmarks[UPPER_LIP_MID_RIGHT];
        const p267 = landmarks[UPPER_LIP_MID_LEFT];
        const p13 = landmarks[UPPER_LIP_INNER];
        const p14 = landmarks[LOWER_LIP_INNER];
        const p17 = landmarks[LOWER_LIP_CENTER];
        const p84 = landmarks[LOWER_LIP_MID_RIGHT];
        const p314 = landmarks[LOWER_LIP_MID_LEFT];

        // 1. Face Width & Mouth Width Normalization
        let faceWidth = euclideanDist(p33, p263);
        if (faceWidth < 1e-6) faceWidth = 1.0;

        let mouthWidth = euclideanDist(p61, p291);
        if (mouthWidth < 1e-6) mouthWidth = 1.0;

        const apertureDist = euclideanDist(p13, p14);

        // 2. Compute 8 Kinematic Features (f1 to f8)
        const f1 = apertureDist / faceWidth;
        const f2 = mouthWidth / faceWidth;
        const upperMidYAvg = ((p37.y || 0) + (p267.y || 0)) / 2.0;
        const f3 = ((p0.y || 0) - upperMidYAvg) / mouthWidth;

        const lowerMidYAvg = ((p84.y || 0) + (p314.y || 0)) / 2.0;
        const f4 = ((p17.y || 0) - lowerMidYAvg) / mouthWidth;

        const f5 = ((p61.y || 0) - (p291.y || 0)) / mouthWidth;

        const f6 = prevF1Ref.current !== null ? (f1 - prevF1Ref.current) : 0.0;
        const f7 = prevF2Ref.current !== null ? (f2 - prevF2Ref.current) : 0.0;
        const f8 = prevF6Ref.current !== null ? (f6 - prevF6Ref.current) : 0.0;

        prevF1Ref.current = f1;
        prevF2Ref.current = f2;
        prevF6Ref.current = f6;

        const vector = [
          Number(f1.toFixed(5)),
          Number(f2.toFixed(5)),
          Number(f3.toFixed(5)),
          Number(f4.toFixed(5)),
          Number(f5.toFixed(5)),
          Number(f6.toFixed(5)),
          Number(f7.toFixed(5)),
          Number(f8.toFixed(5))
        ];

        if (isCapturing && onFrameVectorCalculated) {
          onFrameVectorCalculated(vector);
        }

        // 3. Render Real MediaPipe 3D Mesh Overlay
        ctx.strokeStyle = '#06b6d4';
        ctx.fillStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.shadowBlur = 8;
        ctx.shadowColor = '#06b6d4';

        // Draw Outer Lip Contour
        ctx.beginPath();
        OUTER_LIP_PATH.forEach((idx, i) => {
          const pt = landmarks[idx];
          const x = pt.x * canvas.width;
          const y = pt.y * canvas.height;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();

        // Draw Inner Lip Contour
        ctx.beginPath();
        INNER_LIP_PATH.forEach((idx, i) => {
          const pt = landmarks[idx];
          const x = pt.x * canvas.width;
          const y = pt.y * canvas.height;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();

        // Draw Anchor Nodes
        [p33, p263, p61, p291, p0, p17, p13, p14].forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    } else {
      setFaceDetected(false);
    }

    ctx.restore();
  }, [isCapturing, onFrameVectorCalculated]);

  useEffect(() => {
    let active = true;
    let cameraInstance = null;

    async function initMediaPipe() {
      try {
        const FaceMeshClass = window.FaceMesh || (await import('@mediapipe/face_mesh')).FaceMesh;
        const faceMesh = new FaceMeshClass({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        faceMesh.onResults((results) => {
          if (active) handleFaceMeshResults(results);
        });

        faceMeshRef.current = faceMesh;

        if (videoRef.current) {
          const CameraClass = window.Camera || (await import('@mediapipe/camera_utils')).Camera;
          cameraInstance = new CameraClass(videoRef.current, {
            onFrame: async () => {
              if (active && faceMeshRef.current && videoRef.current) {
                await faceMeshRef.current.send({ image: videoRef.current });
              }
            },
            width: 640,
            height: 480
          });

          await cameraInstance.start();
          if (active) {
            setCameraActive(true);
            setCameraError(null);
          }
          cameraRef.current = cameraInstance;
        }
      } catch (err) {
        console.error('MediaPipe initialization error:', err);
        // Direct webcam fallback
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
          if (videoRef.current && active) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            setCameraActive(true);
          }
        } catch (camErr) {
          setCameraError('Camera access denied or unavailable.');
        }
      }
    }

    initMediaPipe();

    return () => {
      active = false;
      if (cameraRef.current) {
        try { cameraRef.current.stop(); } catch (e) {}
      }
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [handleFaceMeshResults]);

  return (
    <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-slate-900 border border-slate-800 shadow-2xl group">
      
      {/* Hidden Video element for MediaPipe stream */}
      <video ref={videoRef} className="hidden" playsInline muted />

      {/* HTML5 Canvas with Real Landmarked Mesh */}
      <canvas ref={canvasRef} className="w-full h-full object-cover transform -scale-x-100" />

      {/* Error Alert */}
      {cameraError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 text-rose-400 p-6 text-center">
          <CameraIcon className="w-12 h-12 mb-3 text-rose-500 animate-bounce" />
          <p className="font-semibold text-sm">{cameraError}</p>
          <p className="text-xs text-slate-500 mt-1">Please allow webcam access in your browser.</p>
        </div>
      )}

      {/* Live MediaPipe Mesh Status Badge */}
      <div className="absolute top-4 left-4 flex items-center space-x-2 bg-slate-950/70 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-700/60 text-xs text-slate-200">
        <Scan className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
        <span className="font-mono">
          MEDIAPIPE 468 MESH {fps > 0 ? `${fps} FPS` : 'CONNECTING...'}
        </span>
      </div>

      {/* Face Tracking Indicator */}
      <div className="absolute top-4 right-4 flex items-center space-x-2">
        <div className={`px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-md border ${
          faceDetected ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
        }`}>
          {faceDetected ? 'Face Locked' : 'Searching Face...'}
        </div>

        <button
          onClick={() => setIsCapturing(!isCapturing)}
          className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-md border transition-all ${
            isCapturing
              ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50 glow-cyan'
              : 'bg-slate-800/80 text-slate-400 border-slate-700'
          }`}
        >
          <Eye className={`w-3.5 h-3.5 ${isCapturing ? 'animate-pulse text-cyan-400' : ''}`} />
          <span>{isCapturing ? 'Active' : 'Paused'}</span>
        </button>
      </div>

      {/* Bottom Visualizer Bar */}
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-slate-950 via-slate-950/60 to-transparent p-4 flex items-center justify-between">
        <div className="text-xs text-slate-400">
          <span className="text-cyan-400 font-semibold">Kinematics:</span> 8-D Real-Time Lip Coordinate Extraction
        </div>
        <div className="flex space-x-1">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="w-1 bg-cyan-400 rounded-full animate-pulse"
              style={{
                height: `${12 + (i % 3) * 6}px`,
                animationDelay: `${i * 0.15}s`
              }}
            />
          ))}
        </div>
      </div>

    </div>
  );
}
