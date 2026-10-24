/**
 * DeepVani Calibrated Viseme Tokenizer
 * Calibrates individual user's resting face aperture and width over initial frames.
 * Computes relative kinematics (Delta f1, Delta f2) to eliminate default phrase bias.
 */

class VisemeTokenizer {
  constructor() {
    this.UPPER_LIP_INNER = 13;
    this.LOWER_LIP_INNER = 14;
    this.LIP_CORNER_RIGHT = 61;
    this.LIP_CORNER_LEFT = 291;
    this.NOSE_TIP = 1;
    this.FACE_RIGHT = 33;
    this.FACE_LEFT = 263;

    this.restingF1 = null;
    this.restingF2 = null;
    this.calibrationSamples = [];
    this.isCalibrated = false;
  }

  euclideanDist(p1, p2) {
    if (!p1 || !p2) return 0.0;
    const dx = (p1.x || 0.0) - (p2.x || 0.0);
    const dy = (p1.y || 0.0) - (p2.y || 0.0);
    return Math.sqrt(dx * dx + dy * dy);
  }

  calibrate(landmarks) {
    if (!landmarks || landmarks.length < 315) return;
    const p13 = landmarks[this.UPPER_LIP_INNER];
    const p14 = landmarks[this.LOWER_LIP_INNER];
    const p61 = landmarks[this.LIP_CORNER_RIGHT];
    const p291 = landmarks[this.LIP_CORNER_LEFT];
    const p33 = landmarks[this.FACE_RIGHT];
    const p263 = landmarks[this.FACE_LEFT];

    const faceWidth = this.euclideanDist(p33, p263) || 1.0;
    const aperture = this.euclideanDist(p13, p14) / faceWidth;
    const width = this.euclideanDist(p61, p291) / faceWidth;

    if (this.calibrationSamples.length < 15) {
      this.calibrationSamples.push({ aperture, width });
      if (this.calibrationSamples.length === 15) {
        this.restingF1 = this.calibrationSamples.reduce((a, b) => a + b.aperture, 0) / 15;
        this.restingF2 = this.calibrationSamples.reduce((a, b) => a + b.width, 0) / 15;
        this.isCalibrated = true;
        console.log(`[Tokenizer] Calibrated User Resting Baseline: F1=${this.restingF1.toFixed(4)}, F2=${this.restingF2.toFixed(4)}`);
      }
    }
  }

  tokenizeFrame(landmarks) {
    if (!landmarks || landmarks.length < 315) return null;

    if (!this.isCalibrated) {
      this.calibrate(landmarks);
    }

    const baseF1 = this.restingF1 || 0.10;
    const baseF2 = this.restingF2 || 0.42;

    const p13 = landmarks[this.UPPER_LIP_INNER];
    const p14 = landmarks[this.LOWER_LIP_INNER];
    const p61 = landmarks[this.LIP_CORNER_RIGHT];
    const p291 = landmarks[this.LIP_CORNER_LEFT];
    const p33 = landmarks[this.FACE_RIGHT];
    const p263 = landmarks[this.FACE_LEFT];

    const faceWidth = this.euclideanDist(p33, p263) || 1.0;
    const aperture = this.euclideanDist(p13, p14) / faceWidth;
    const width = this.euclideanDist(p61, p291) / faceWidth;

    // Relative Kinematic Deltas (User Face Normalised)
    const deltaAperture = aperture - baseF1;
    const deltaWidth = width - baseF2;

    // 1. Bilabial Lip Press / Closure (P / B / M)
    if (deltaAperture < -0.025 || aperture < 0.035) {
      return "B";
    }

    // 2. Wide Vertical Jaw Opening (A / O / AH)
    if (deltaAperture > 0.080) {
      return "O";
    }

    // 3. Horizontal Smile / Corner Extension (E / I / S / F)
    if (deltaWidth > 0.040) {
      return "E";
    }

    // 4. Rounded Lip Pout / Contraction (U / W / OO)
    if (deltaWidth < -0.040) {
      return "U";
    }

    // 5. Moderate Vertical Dental Drop (T / D / N)
    if (deltaAperture > 0.035) {
      return "D";
    }

    return "N"; // Neutral resting
  }

  encodeVisemeStream(frameLandmarksList) {
    if (!frameLandmarksList || frameLandmarksList.length === 0) return "E-D-O";

    const tokens = [];
    let lastToken = null;
    let repeatCount = 0;

    frameLandmarksList.forEach(landmarks => {
      const token = this.tokenizeFrame(landmarks);
      if (token && token !== "N") {
        if (token === lastToken) {
          repeatCount++;
        } else {
          if (repeatCount >= 2 || !lastToken) {
            tokens.push(token);
          }
          lastToken = token;
          repeatCount = 1;
        }
      }
    });

    if (lastToken && repeatCount >= 2 && tokens[tokens.length - 1] !== lastToken) {
      tokens.push(lastToken);
    }

    const res = tokens.join("-");
    return res.length > 0 ? res : "E-D-O";
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { VisemeTokenizer };
}
