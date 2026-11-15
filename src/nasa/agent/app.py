import json
import logging
import os
from contextlib import asynccontextmanager

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import meta_cache
from .advice import fetch_advice
from .extractor import extract
from .models import ExtractionResponse, HealthResponse, TextRequest

load_dotenv()

log = logging.getLogger("nasa.agent")

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB — Whisper limit
SUPPORTED_TYPES = {
    "audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav",
    "audio/webm", "audio/ogg", "audio/flac", "video/mp4",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_url = os.getenv("NASA_API_BASE_URL", "https://nasa-api-miva-pulsars.duckdns.org/api/v1")
    meta_cache.load(base_url)
    meta_cache.start_background_refresh(base_url)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="NASA Field Shift — Voice Extraction Service",
        version="0.1.0",
        description="Converts a farmer's spoken message into structured JSON for POST /api/v1/advice.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", meta_loaded=meta_cache.is_loaded())

    @app.post("/extract/text", response_model=ExtractionResponse)
    def extract_text(body: TextRequest) -> ExtractionResponse:
        meta = meta_cache.get()
        result = extract(transcript=body.message, meta=meta, previous=body.context)
        if result.confidence == "complete":
            result = fetch_advice(result)
        return result

    @app.post("/extract/voice", response_model=ExtractionResponse)
    async def extract_voice(
        audio: UploadFile = File(...),
        state: str | None = Form(None),
        lga: str | None = Form(None),
        context: str | None = Form(None),  # JSON string of previous ExtractionResponse
    ) -> ExtractionResponse:
        previous = None
        if context:
            try:
                previous = json.loads(context)
            except json.JSONDecodeError:
                pass
        content_type = audio.content_type or ""
        if content_type not in SUPPORTED_TYPES and not content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail=f"Unsupported audio type: {content_type}")

        audio_bytes = await audio.read()
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 25 MB limit")

        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=(audio.filename or "audio", audio_bytes, content_type),
                language="en",
            )
            transcript = result.text
        except Exception as exc:
            log.error("Whisper transcription failed: %s", exc)
            raise HTTPException(status_code=502, detail="Audio transcription failed. Please try again.") from exc

        meta = meta_cache.get()
        result = extract(transcript=transcript, meta=meta, hint_state=state, hint_lga=lga, previous=previous)
        if result.confidence == "complete":
            result = fetch_advice(result)
        return result

    return app


app = create_app()
