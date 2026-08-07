"""macos-stt CLI — macOS system speech-to-text."""

from __future__ import annotations

import json
import sys

import typer

from .engine import (
    MacOSSpeechError,
    list_languages,
    transcribe_file,
    transcribe_bytes,
)

app = typer.Typer(
    name="macos-stt",
    help="macOS 系统语音转文字 — 零配置，全离线，63 种语言",
)


@app.command()
def transcribe(
    file: str = typer.Argument(..., help="音频文件路径"),
    lang: str = typer.Option("zh-CN", "--lang", "-l", help="语言代码"),
    json_output: bool = typer.Option(False, "--json", help="JSON 格式输出"),
    timeout: float = typer.Option(15.0, "--timeout", help="超时秒数"),
):
    """识别音频文件。支持 WAV、MP3、M4A 等所有 macOS 原生音频格式。"""
    try:
        result = transcribe_file(file, language=lang, timeout=timeout)
        if json_output:
            typer.echo(json.dumps({
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
            }, ensure_ascii=False))
        else:
            typer.echo(result.text if result.text else "")
    except MacOSSpeechError as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def raw(
    lang: str = typer.Option("zh-CN", "--lang", "-l", help="语言代码"),
    sr: int = typer.Option(16000, "--sr", help="PCM 采样率"),
    timeout: float = typer.Option(15.0, "--timeout", help="超时秒数"),
):
    """从 stdin 读取 PCM 字节流并识别。"""
    try:
        data = sys.stdin.buffer.read()
        result = transcribe_bytes(data, sample_rate=sr, language=lang, timeout=timeout)
        typer.echo(result.text if result.text else "")
    except MacOSSpeechError as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="list-languages")
def list_languages_cmd():
    """列出所有 63 种支持的语言。"""
    for lang in list_languages():
        typer.echo(lang)


@app.command()
def serve(port: int = typer.Option(8765, help="HTTP 服务端口")):
    """启动 HTTP 识别服务（需要 pip install macos-stt[server]）。"""
    try:
        from macos_stt.server import create_app
        import uvicorn
    except ImportError:
        typer.echo("HTTP 服务需要: pip install macos-stt[server]", err=True)
        raise typer.Exit(1)

    http_app = create_app()
    typer.echo(f"macos-stt HTTP server → http://127.0.0.1:{port}")
    uvicorn.run(http_app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
