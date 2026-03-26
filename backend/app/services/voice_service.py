import logging
import asyncio
import base64

import google.generativeai as genai
import httpx

from app.config import app_config

logger = logging.getLogger(__name__)


class VoiceService:
    def _is_gemini_voice_model(self) -> bool:
        model_name = (app_config.voice_model or "").lower()
        return "gemini" in model_name

    def _audio_format_from_mime(self, mime_type: str) -> str:
        normalized = (mime_type or "audio/ogg").lower().strip()
        mapping = {
            "audio/ogg": "ogg",
            "audio/oga": "ogg",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/flac": "flac",
            "audio/aac": "aac",
            "audio/mp4": "m4a",
            "audio/x-m4a": "m4a",
            "audio/webm": "webm",
        }
        return mapping.get(normalized, "ogg")

    def _extract_openai_style_text(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item["text"]))
                    elif item.get("text"):
                        parts.append(str(item["text"]))
            return "\n".join(p for p in parts if p).strip()
        return ""

    async def _transcribe_via_openai_compatible(self, audio_bytes: bytes, mime_type: str) -> str:
        api_key = app_config.voice_api_key or app_config.llm_api_key
        if not api_key:
            raise RuntimeError("VOICE_API_KEY or LLM_API_KEY is not configured")

        base_url = (app_config.voice_base_url or app_config.llm_base_url or "").rstrip("/")
        if not base_url:
            raise RuntimeError("VOICE_BASE_URL is not configured")

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "model": app_config.voice_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio exactly. Return only the spoken words without extra commentary.",
                        },
                        {
                            "type": "input_audio",
                            "inputAudio": {
                                "data": audio_b64,
                                "format": self._audio_format_from_mime(mime_type),
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 200,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **app_config.llm_default_headers,
        }
        endpoint = f"{base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
        if response.status_code >= 400:
            detail = response.text[:500]
            lowered = detail.lower()
            if "audio" in lowered and ("not support" in lowered or "unsupported" in lowered or "invalid" in lowered):
                raise RuntimeError(
                    "The configured `VOICE_MODEL` does not support audio input on your provider. "
                    "Set `VOICE_MODEL` to an audio-capable OpenRouter model or a Gemini model."
                )
            raise RuntimeError(f"Voice transcription request failed: {response.status_code} {detail}")

        text = self._extract_openai_style_text(response.json())
        if not text:
            raise RuntimeError("Voice transcription returned an empty transcript")
        return text

    async def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        api_key = app_config.llm_api_key or app_config.voice_api_key
        if not api_key:
            raise RuntimeError("LLM_API_KEY is not configured")

        if not audio_bytes:
            raise RuntimeError("Voice payload is empty")

        if not self._is_gemini_voice_model():
            return await self._transcribe_via_openai_compatible(audio_bytes, mime_type)

        # Keep one-key setup: Gemini handles both chat + voice.
        genai.configure(api_key=api_key)
        model_name = app_config.voice_model or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name=model_name)

        prompt = (
            "Transcribe this audio exactly. "
            "Return only the spoken text without extra commentary."
        )
        content = [
            prompt,
            {
                "mime_type": mime_type or "audio/ogg",
                "data": audio_bytes,
            },
        ]

        response = await asyncio.to_thread(model.generate_content, content)
        text = (getattr(response, "text", "") or "").strip()

        if not text and hasattr(response, "candidates"):
            parts = []
            for candidate in getattr(response, "candidates", []) or []:
                candidate_content = getattr(candidate, "content", None)
                for part in (getattr(candidate_content, "parts", None) or []):
                    ptext = getattr(part, "text", None)
                    if ptext:
                        parts.append(ptext)
            text = "\n".join(parts).strip()

        if not text:
            raise RuntimeError("Gemini returned an empty transcript")

        return text

    async def transcribe_ogg(self, audio_bytes: bytes) -> str:
        """Backward-compatible wrapper."""
        return await self.transcribe_audio(audio_bytes, mime_type="audio/ogg")


voice_service = VoiceService()
