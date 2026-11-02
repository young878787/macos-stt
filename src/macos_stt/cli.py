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
    help="macOS 系統語音轉文字 — 零配置，全離線，63 種語言",
)


@app.command()
def transcribe(
    file: str = typer.Argument(..., help="音訊檔案路徑"),
    lang: str = typer.Option("zh-CN", "--lang", "-l", help="語言代碼"),
    json_output: bool = typer.Option(False, "--json", help="JSON 格式輸出"),
    timeout: float = typer.Option(15.0, "--timeout", help="逾時秒數"),
):
    """辨識音訊檔案。支援 WAV、MP3、M4A 等所有 macOS 原生音訊格式。"""
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
        typer.echo(f"錯誤: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def raw(
    lang: str = typer.Option("zh-CN", "--lang", "-l", help="語言代碼"),
    sr: int = typer.Option(16000, "--sr", help="PCM 取樣率"),
    timeout: float = typer.Option(15.0, "--timeout", help="逾時秒數"),
):
    """從 stdin 讀取 PCM 位元組流並辨識。"""
    try:
        data = sys.stdin.buffer.read()
        result = transcribe_bytes(data, sample_rate=sr, language=lang, timeout=timeout)
        typer.echo(result.text if result.text else "")
    except MacOSSpeechError as e:
        typer.echo(f"錯誤: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="list-languages")
def list_languages_cmd():
    """列出所有 63 種支援的語言。"""
    for lang in list_languages():
        typer.echo(lang)


@app.command()
def serve(
    port: int = typer.Option(8765, help="HTTP 服務連接埠（僅本機 127.0.0.1）"),
    lang: str = typer.Option("zh-TW", "--lang", "-l", help="全域預設語言"),
    vocab: str = typer.Option(None, "--vocab", help="詞表檔（每行/逗號分隔）；預設內建 vocab.default.txt"),
    max_sessions: int = typer.Option(3, "--max-sessions", help="最大同時串流 session 數"),
    allow_cloud: bool = typer.Option(False, "--allow-cloud", help="允許 on_device=false（雲端辨識）"),
):
    """啟動 HTTP/WS 辨識服務（需要 pip install macos-stt[server]）。

    端點：REST /transcribe、/transcribe/raw、/vocabulary、/sessions、
    /health、/languages；串流 WS /stream（各專案把 PCM 推上來即時收文字）。
    """
    try:
        from macos_stt.server import create_app
        from macos_stt.vocab import VocabularyStore
        import uvicorn
    except ImportError:
        typer.echo("HTTP 服務需要: pip install macos-stt[server]", err=True)
        raise typer.Exit(1)

    try:
        store = VocabularyStore.load_file(vocab) if vocab else VocabularyStore.load_default()
        http_app = create_app(
            default_lang=lang, vocab=store, max_sessions=max_sessions, allow_cloud=allow_cloud
        )
    except FileNotFoundError:
        typer.echo(f"詞表檔不存在: {vocab}", err=True)
        raise typer.Exit(1)

    # 授權狀態檢查（只讀，不呼叫 requestAuthorization_——非 .app bundle 環境
    # 主動 request 會觸發 TCC SIGABRT）。未授權時建議改用 start.py serve。
    from macos_stt import engine as _engine
    from Speech import (
        SFSpeechRecognizer,
        SFSpeechRecognizerAuthorizationStatusAuthorized,
    )
    status = int(SFSpeechRecognizer.authorizationStatus())
    if status != int(SFSpeechRecognizerAuthorizationStatusAuthorized):
        typer.echo(
            "警告: 語音辨識未授權（本啟動方式無法彈授權窗）。"
            "建議改用 `python start.py serve`（MacSTT.app bundle，首次運行會彈授權對話框）；"
            "或到 系統設定 → 隱私權與安全性 → 語音辨識 手動開啟。",
            err=True,
        )

    typer.echo(
        f"macos-stt server → http://127.0.0.1:{port} (WS /stream) | "
        f"lang={lang} sessions<={max_sessions} vocab={len(store)}"
    )
    uvicorn.run(http_app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    app()
