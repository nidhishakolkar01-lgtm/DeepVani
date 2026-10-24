"""
DeepVani Agent Service Proxy
Delegates open-vocabulary viseme stream decoding and sentence rescoring to master_ml_engine in backend/ml/ml_engine.py.
"""

from typing import Optional, Dict, Any
from backend.ml.ml_engine import master_ml_engine


class AgentDisambiguatorService:
    """
    Service Proxy delegating all ML and LLM decoding (LM Studio, Ollama, Groq) to master_ml_engine.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.ml_engine = master_ml_engine

    async def rescore_viseme_stream_async(self, viseme_stream: str, language: str = "auto") -> str:
        res = await self.ml_engine.decode_viseme_stream_async(viseme_stream, language)
        return res.get("candidate", "Namaste")

    async def polish_sentence_async(self, candidate_text: str, language: str = "hi") -> str:
        return candidate_text
