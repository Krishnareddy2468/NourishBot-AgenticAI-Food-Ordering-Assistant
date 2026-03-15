import io
import logging

from openai import AsyncOpenAI

from app.config import app_config

logger = logging.getLogger(__name__)


class VoiceService:
    async def transcribe_ogg(self, audio_bytes: bytes) -> str:
        if not app_config.voice_api_key:
            raise RuntimeError("VOICE_API_KEY is not configured")

        client = AsyncOpenAI(
            api_key=app_config.voice_api_key,
            base_url=app_config.voice_base_url,
        )
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "voice.ogg"
        transcript = await client.audio.transcriptions.create(
            model=app_config.voice_model,
            file=audio_file,
        )
        text = getattr(transcript, "text", "") or ""
        if not text.strip():
            raise RuntimeError("Empty transcript from voice service")
        return text.strip()


voice_service = VoiceService()
