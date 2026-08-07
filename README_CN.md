# macos-stt

**macOS 系统语音识别 — 零配置，全离线，63 种语言。**

[![PyPI](https://img.shields.io/pypi/v/macos-stt)](https://pypi.org/project/macos-stt/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)](https://apple.com/macos)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **English docs**: [README.md](README.md)

每台 Mac 里都内置了一个强大的本地语音识别引擎——Siri 用它，听写也用它。Apple 没把它暴露给 Python 开发者。`macos-stt` 把这个引擎做成了 `pip install` 就能用的库。**不下载模型、不注册 API Key、不联网**。

## 为什么会做这个

macOS 上所有 Python 语音识别方案——whisper、faster-whisper、mlx-whisper——都要下载几个 G 的模型。但 `SFSpeechRecognizer` 就躺在 `/System/Library/Frameworks/Speech.framework` 里。买 Mac 的钱已经付过这台引擎了，只是没人把它做成 Python 库。

## 特点

- **零下载** — 不下载模型，不注册 Key，不签协议
- **全离线** — 飞行模式也能用，数据不离开本机
- **63 种语言** — 含中文、粤语、日语、英语等
- **三种接口** — Python 库、CLI、HTTP 服务
- **快** — M1 Pro 上 3 秒音频约 300ms 完成（ANE 加速）

## 安装

```bash
pip install macos-stt
# 仅支持 macOS，Python 3.9+

# 可选：HTTP 服务
pip install macos-stt[server]
```

## 快速开始

### Python 库

```python
from macos_stt import recognize, recognize_bytes, list_languages

# 63 种语言
print(len(list_languages()))  # 63

# 识别音频文件（WAV、MP3、M4A 等 macOS 原生支持的所有格式）
text = recognize("录音.wav", language="zh-CN")
print(text)

# 识别 PCM 字节流
text = recognize_bytes(pcm_bytes, sample_rate=16000, language="zh-CN")
```

### CLI

```bash
macos-stt transcribe 录音.mp3 --lang zh-CN
macos-stt raw --sr 16000 < audio.pcm
macos-stt list-languages
macos-stt serve --port 8765
```

### HTTP 服务

```bash
macos-stt serve
# POST /transcribe     — 文件上传识别
# POST /transcribe/raw — base64 PCM 字节流
# GET  /languages      — 63 种语言
# GET  /health         — 健康检查
```

## 性能

| | macos-stt | whisper (MLX) | faster-whisper |
|---|---|---|---|
| 3 秒音频 (M1 Pro) | **~300ms** | ~500ms | ~800ms |
| 模型下载 | **0** | 140 MB | 1.5 GB |
| 内存占用 | **可忽略** | ~500 MB | ~1 GB |
| 支持语言 | 63 | ~100 | ~100 |

## 跟其他方案对比

| 库 | 引擎 | 要下载模型？ | 离线？ |
|---|---|---|---|
| **macos-stt** | macOS 系统 | **0** | ✅ |
| `openai-whisper` | Whisper | 1-3 GB | ✅ |
| `faster-whisper` | Whisper (CTranslate2) | 1-3 GB | ✅ |
| `mlx-whisper` | Whisper (MLX) | 1-3 GB | ✅ |
| `whisper.cpp` | Whisper (C++) | 1-3 GB | ✅ |

所有其他方案都要下载模型。`macos-stt` 用的是 Mac 上已有的引擎。

## 原理

通过 PyObjC 桥接 Apple 的 `SFSpeechRecognizer`。核心踩坑：

- **RunLoop 线程管理** — SFSpeechRecognizer 的回调必须由主线程 CFRunLoop 派发。用 daemon 线程 + `runMode:beforeDate:` + `threading.Event` 同步解决
- **授权流程** — `requestAuthorization:` 是异步回调，必须在等待授权结果的同时 Pump RunLoop
- **PyObjC 枚举问题** — 授权常量是扁平的 `NewType` 全局变量，不是 enum 属性

## 后续计划

`macos-stt` 是 `macos-ml` 家族的第一个项目：

- **macos-audio** — `SNAudioClassifier`，425+ 种声音分类（狗叫、婴儿哭、玻璃碎…）
- **macos-embed** — `NLEmbedding`，7 种语言词向量，零下载
- **macos-vision** — Vision 框架，OCR、人脸检测、图像相似度

同一套哲学：零下载，用 Mac 上已有的 ML 引擎。

## License

MIT.
