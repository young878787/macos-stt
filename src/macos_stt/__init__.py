"""macos-stt: macOS system speech recognition — zero config, fully offline."""

from .engine import (
    AuthorizationError,
    list_languages,
    MacOSSpeechError,
    RecognitionError,
    RecognitionResult,
    RecognitionTimeout,
    transcribe_bytes,
    transcribe_file,
    UnsupportedLanguage,
)

recognize = transcribe_file
recognize_bytes = transcribe_bytes

__all__ = [
    "recognize",
    "recognize_bytes",
    "list_languages",
    "RecognitionResult",
    "MacOSSpeechError",
    "AuthorizationError",
    "RecognitionError",
    "RecognitionTimeout",
    "UnsupportedLanguage",
]
