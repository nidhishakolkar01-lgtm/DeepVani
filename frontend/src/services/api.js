import axios from 'axios';

const API_BASE = '/api/v1';
const DEFAULT_API_KEY = 'deepvani-local-dev-key-2026';

// Configure Axios Defaults with Security Header
const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': DEFAULT_API_KEY
  }
});

export const checkHealth = async () => {
  try {
    const res = await apiClient.get('/health');
    return res.data;
  } catch (err) {
    console.error('Health check failed:', err);
    return { status: 'offline', model_loaded: false };
  }
};

export const fetchPhrases = async () => {
  try {
    const res = await apiClient.get('/phrases');
    return res.data.phrases || [];
  } catch (err) {
    console.error('Failed to fetch phrases:', err);
    return [];
  }
};

export const decodeSilentSpeech = async (framesVector, language = 'hi') => {
  try {
    const res = await apiClient.post('/decode-silent-speech', {
      language,
      frames_vector: framesVector
    });
    return res.data;
  } catch (err) {
    console.error('Decode silent speech error:', err);
    return { status: 'error', message: err.message };
  }
};

export const quickTrainPhrase = async (phraseLabel, sampleTensors, language = 'hi') => {
  try {
    const res = await apiClient.post('/quick-train', {
      phrase_label: phraseLabel,
      language,
      sample_tensors: sampleTensors
    });
    return res.data;
  } catch (err) {
    console.error('Quick train error:', err);
    return { status: 'error', message: err.message };
  }
};

/**
 * Creates a persistent duplex WebSocket connection for real-time streaming
 */
export const createLiveStreamSocket = (onRecognitionEvent, onStatusChange) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.port === '5173' ? 'localhost:8000' : window.location.host;
  const wsUrl = `${protocol}//${host}/ws/live-stream`;

  let ws = null;
  let isClosedIntentionally = false;

  const connect = () => {
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        if (onStatusChange) onStatusChange('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'RECOGNITION_EVENT' && onRecognitionEvent) {
            onRecognitionEvent(data);
          }
        } catch (e) {
          console.error('WS parse error:', e);
        }
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error (will fallback to REST):', err);
        if (onStatusChange) onStatusChange('error');
      };

      ws.onclose = () => {
        if (onStatusChange) onStatusChange('disconnected');
        if (!isClosedIntentionally) {
          setTimeout(connect, 3000); // Auto-reconnect after 3 seconds
        }
      };
    } catch (e) {
      console.warn('Failed to establish WebSocket:', e);
      if (onStatusChange) onStatusChange('failed');
    }
  };

  connect();

  return {
    sendVector: (vector, language = 'hi') => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ vector, language }));
        return true;
      }
      return false;
    },
    close: () => {
      isClosedIntentionally = true;
      if (ws) ws.close();
    }
  };
};
