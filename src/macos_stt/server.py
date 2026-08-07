"""HTTP server for macos-stt."""

from __future__ import annotations

import base64
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .engine import (
    MacOSSpeechError,
    RecognitionTimeout,
    UnsupportedLanguage,
    list_languages,
    transcribe_file,
    transcribe_bytes,
)


class RawRequest(BaseModel):
    bytes: str
    sr: int = 16000
    lang: str = "zh-CN"
    timeout: float = 15.0


class TranscribeResponse(BaseModel):
    text: str
    confidence: float
    language: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="macos-stt",
        description="macOS 系统语音转文字 HTTP 服务 — 全离线，63 种语言",
        version="0.2.0",
    )

    @app.get("/languages")
    async def get_languages():
        return {"languages": list_languages(), "count": len(list_languages())}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe(file: UploadFile = File(...), lang: str = "zh-CN", timeout: float = 15.0):
        if lang not in list_languages():
            raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'.")

        suffix = ".wav"
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(await file.read())
            tmp.close()
            result = transcribe_file(tmp.name, language=lang, timeout=timeout)
            return TranscribeResponse(text=result.text, confidence=result.confidence, language=result.language)
        except UnsupportedLanguage as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RecognitionTimeout:
            raise HTTPException(status_code=408, detail="Recognition timed out")
        except MacOSSpeechError as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @app.post("/transcribe/raw", response_model=TranscribeResponse)
    async def transcribe_raw(req: RawRequest):
        if req.lang not in list_languages():
            raise HTTPException(status_code=400, detail=f"Unsupported language '{req.lang}'.")
        try:
            data = base64.b64decode(req.bytes)
            result = transcribe_bytes(data, sample_rate=req.sr, language=req.lang, timeout=req.timeout)
            return TranscribeResponse(text=result.text, confidence=result.confidence, language=result.language)
        except UnsupportedLanguage as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RecognitionTimeout:
            raise HTTPException(status_code=408, detail="Recognition timed out")
        except MacOSSpeechError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8765)
