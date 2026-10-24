"""
DeepVani Services - Edge-TTS Neural Speech Synthesizer Module
Synthesizes natural, high-fidelity neural audio from text using Microsoft Azure voices via edge-tts.
Output: MP3 files stored in backend/static/audio/ with UUID filenames and automatic cleanup.
"""

import os
import time
import uuid
import asyncio
from typing import Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class TTSService:
    """
    Neural Text-to-Speech Engine using edge-tts with automated cleanup and unpredictable filenames.
    Default Voices:
    - Hindi: hi-IN-SwaraNeural
    - English: en-IN-NeerjaNeural
    """

    VOICE_MAP = {
        "hi": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural"
    }

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "audio")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _cleanup_old_audio_files(self, max_files: int = 50, max_age_seconds: int = 600):
        """
        Cleans up generated audio files older than max_age_seconds or exceeds max_files.
        """
        try:
            now = time.time()
            files = []
            for f in os.listdir(self.output_dir):
                if f.endswith('.mp3'):
                    fp = os.path.join(self.output_dir, f)
                    try:
                        mtime = os.path.getmtime(fp)
                        files.append((fp, mtime))
                    except OSError:
                        pass

            files.sort(key=lambda x: x[1])

            # Delete files older than max_age_seconds
            for fp, mtime in files:
                if now - mtime > max_age_seconds:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass

            # Prune if more than max_files
            remaining = [f for f in os.listdir(self.output_dir) if f.endswith('.mp3')]
            if len(remaining) > max_files:
                for fp, _ in files[:len(remaining) - max_files]:
                    try:
                        if os.path.exists(fp):
                            os.remove(fp)
                    except OSError:
                        pass
        except Exception as e:
            print(f"Audio cleanup error: {e}")

    async def generate_speech_async(self, text: str, lang: str = "hi") -> Optional[str]:
        """
        Asynchronously generates MP3 file with random UUID token using edge-tts.
        """
        if not text or not text.strip():
            return None

        # Run light cleanup
        self._cleanup_old_audio_files()

        voice = self.VOICE_MAP.get(lang, "hi-IN-SwaraNeural")
        unique_token = uuid.uuid4().hex[:16]
        filename = f"speech_{unique_token}.mp3"
        filepath = os.path.join(self.output_dir, filename)

        if EDGE_TTS_AVAILABLE:
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(filepath)
                return f"/static/audio/{filename}"
            except Exception as e:
                print(f"Edge-TTS synthesis error: {e}")

        return None

    def generate_speech(self, text: str, lang: str = "hi") -> Optional[str]:
        """
        Synchronous wrapper for generate_speech_async.
        """
        try:
            return asyncio.run(self.generate_speech_async(text, lang))
        except Exception as e:
            print(f"Sync TTS execution error: {e}")
            return None
