# macos-stt

**Use your Mac's built-in speech recognizer from Python — zero config, fully offline, 63 languages.**

[![PyPI](https://img.shields.io/pypi/v/macos-stt)](https://pypi.org/project/macos-stt/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)](https://apple.com/macos)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **中文文档**: [README_CN.md](README_CN.md)

Stop downloading 3GB models. Your Mac already has a world-class speech recognizer built in — the same engine that powers Siri and Dictation. `macos-stt` wraps it into a Python library you can `pip install`.

## Why

Every Python speech recognition tool for macOS — whisper, faster-whisper, mlx-whisper — requires downloading a model. Meanwhile, Apple's `SFSpeechRecognizer` sits unused in `/System/Library/Frameworks/Speech.framework`. We paid for this hardware. This library makes the engine accessible.

## Features

- **Zero downloads** — no models, no API keys, no sign-up
- **Fully offline** — works in airplane mode, data never leaves your machine
- **63 languages** — including Chinese (zh-CN, yue-CN), Japanese, English, and more
- **Three interfaces** — Python library, CLI, HTTP server
- **Fast** — ~300ms for 3s audio on M1 Pro (ANE-accelerated)

## Installation

```bash
pip install macos-stt
# macOS only. Python 3.9+.

# Optional: HTTP server support
pip install macos-stt[server]
```

## Quick Start

### Python

```python
from macos_stt import recognize, recognize_bytes, list_languages

# 63 languages
print(len(list_languages()))  # 63

# Transcribe an audio file (WAV, MP3, M4A — anything macOS can decode)
text = recognize("recording.wav", language="zh-CN")
print(text)

# Transcribe raw PCM bytes
text = recognize_bytes(pcm_bytes, sample_rate=16000, language="en-US")
```

### CLI

```bash
macos-stt transcribe recording.mp3 --lang zh-CN
macos-stt raw --sr 16000 < audio.pcm
macos-stt list-languages
macos-stt serve --port 8765
```

### HTTP Server

```bash
macos-stt serve
# POST /transcribe     — file upload
# POST /transcribe/raw — base64 PCM bytes
# GET  /languages      — 63 languages
# GET  /health         — health check
```

## Performance

| | macos-stt | whisper (MLX) | faster-whisper |
|---|---|---|---|
| 3s audio (M1 Pro) | **~300ms** | ~500ms | ~800ms |
| Model download | **0** | 140 MB | 1.5 GB |
| Memory usage | **negligible** | ~500 MB | ~1 GB |
| Languages | 63 | ~100 | ~100 |

## Comparison

| Library | Engine | Model Download | Offline |
|---|---|---|---|
| **macos-stt** | macOS system | **0** | ✅ |
| `openai-whisper` | Whisper | 1-3 GB | ✅ |
| `faster-whisper` | Whisper (CTranslate2) | 1-3 GB | ✅ |
| `mlx-whisper` | Whisper (MLX) | 1-3 GB | ✅ |
| `whisper.cpp` | Whisper (C++) | 1-3 GB | ✅ |

All other options download models. macos-stt uses the engine you already own.

## How It Works

`macos-stt` bridges Apple's `SFSpeechRecognizer` to Python via PyObjC — Apple's own ObjC bridge that ships with macOS. Key technical challenges solved:

- **RunLoop threading** — SFSpeechRecognizer callbacks only fire on the main thread's CFRunLoop. Solved with a daemon thread running `runMode:beforeDate:` + `threading.Event` sync.
- **Authorization flow** — `requestAuthorization:` requires RunLoop pumping during the async permission check.
- **PyObjC enum quirks** — Authorization constants are flat `NewType` globals, not enum attributes.

## Roadmap

`macos-stt` is the first in a planned `macos-ml` family:

- **macos-audio** — `SNAudioClassifier`, 425+ sound categories
- **macos-embed** — `NLEmbedding`, word vectors in 7 languages
- **macos-vision** — OCR, face detection, image similarity

Same philosophy: zero downloads, use the ML engine your Mac already has.

## License

MIT.
