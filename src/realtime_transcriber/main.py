"""CLIエントリポイント.

AudioCapture → transcribe_audio → translate → print のパイプラインを制御する。
VADにより発話区間を自動検出し、自然な文の区切りで処理する。
"""

import argparse
import logging
import re
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import sounddevice as sd

import mlx_whisper

from realtime_transcriber.audio import AudioCapture
from realtime_transcriber.session_logger import SessionLogger
from realtime_transcriber.summarizer import Summarizer
from realtime_transcriber.transcriber import is_hallucination, transcribe_audio
from realtime_transcriber.translator import create_translate_client, translate_text

# --- 設定 ---
DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
LANGUAGE = "en"
SLEEP_SECONDS = 0.05
# 未完結の文を蓄積する最大秒数（これを超えたら未完結でも翻訳に回す）
MAX_PENDING_SECONDS = 15
# prev_text（Whisperコンテキスト）に保持する最大文字数
MAX_CONTEXT_CHARS = 200
# 既知の話者名・固有名詞リスト（Whisperのinitial_promptに含めて認識精度を向上）
# セッションごとに編集して使う。空リストの場合は無視される。
KNOWN_SPEAKERS: list[str] = []

# 翻訳の最大並列数
MAX_TRANSLATE_WORKERS = 5

# --- ANSIエスケープ ---
_DIM = "\033[90m"  # グレー文字（原文表示用）
_YELLOW = "\033[93m"  # 黄色（ステータス表示用）
_RESET = "\033[0m"
_CLEAR_LINE = "\r\033[K"  # カーソル行をクリア

logger = logging.getLogger(__name__)

# --- 文分割ユーティリティ ---

_ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|etc|vs|e\.g|i\.e|al|Gen|Gov|Sgt|Corp)\.",
    re.IGNORECASE,
)
_ABBR_PLACEHOLDER = "\x00"


def _is_sentence_end(text: str) -> bool:
    """テキストが文末（. ! ? など）で終わっているか判定する.

    省略記号 "..." は文末とみなさない。
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith("..."):
        return False
    return stripped[-1] in ".!?;"


def _split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する.

    略語（Mr. Dr. e.g. など）のピリオドでは分割しない。
    """
    # 略語のピリオドをプレースホルダに退避して誤分割を防ぐ
    protected = _ABBREVIATIONS.sub(
        lambda m: m.group()[:-1] + _ABBR_PLACEHOLDER, text.strip()
    )
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    # プレースホルダをピリオドに復元して返す
    return [s.replace(_ABBR_PLACEHOLDER, ".") for s in sentences if s.strip()]


# --- メインループのサブ処理 ---


def _check_audio_output() -> None:
    """起動時に音声出力先を確認し、Multi-Output Deviceでなければ切替を促す."""
    default_out = sd.default.device[1]
    device_name = sd.query_devices(default_out)["name"]
    print(f"Output device: {device_name}")

    if "複数出力" not in device_name and "multi" not in device_name.lower():
        print("⚠ Please switch output to Multi-Output Device.")
        print("  Opening Sound Settings...")
        subprocess.run(
            ["open", "x-apple.systempreferences:com.apple.Sound-Settings.extension"],
        )
        input("  Press Enter after switching: ")


def _build_context(sentences: list[str]) -> str:
    """直近の文から Whisper の initial_prompt 用コンテキストを組み立てる.

    末尾の文から逆順にたどり、MAX_CONTEXT_CHARS 以内に収まる範囲を返す。
    文の途中で切れないようにするため、文単位で取得する。
    """
    parts: list[str] = []
    char_count = 0
    for s in reversed(sentences):
        if char_count + len(s) > MAX_CONTEXT_CHARS:
            break
        parts.append(s)
        char_count += len(s)
    return " ".join(reversed(parts))


def _translate_sentences(
    sentences: list[str],
    translate_client: object,
    session_context: str = "",
) -> list[str]:
    """複数の文を並列で翻訳し、入力と同じ順序で翻訳結果を返す."""

    if not sentences:
        return []

    def _translate_one(s: str) -> str:
        try:
            return translate_text(
                text=s,
                source_lang="en",
                target_lang="ja",
                client=translate_client,
                session_context=session_context,
            )
        except Exception:
            logger.exception("Translation failed")
            return "(翻訳失敗)"

    max_workers = min(len(sentences), MAX_TRANSLATE_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_translate_one, sentences))


def _print_results(
    sentences: list[str],
    translated: list[str],
    session_logger: SessionLogger,
) -> None:
    """原文（グレー）と翻訳文を経過時間付きで表示し、ログに記録する."""
    ts = session_logger.elapsed()
    for i, (sentence, ja_text) in enumerate(zip(sentences, translated)):
        marker = "▸" if i == 0 else " "
        print(f"{_DIM}{marker} {ts} {sentence}{_RESET}", flush=True)
        print(f"{marker} {ts} {ja_text}", flush=True)
        session_logger.log(sentence, ja_text)
    print("", flush=True)


# --- エントリポイント ---


def _process_chunk(
    chunk: np.ndarray,
    initial_prompt: str | None,
    translate_client: object,
    session_logger: SessionLogger,
    summarizer: Summarizer,
    capture: AudioCapture,
    context: dict,
) -> None:
    """1チャンクの文字起こし→翻訳→表示を実行する（ワーカースレッド用）."""
    print(f"\r{_YELLOW}⏳ Transcribing...{_RESET}", end="", flush=True)
    text = transcribe_audio(
        audio=chunk,
        language=LANGUAGE,
        mlx_whisper_module=mlx_whisper,
        initial_prompt=initial_prompt,
    )
    print(_CLEAR_LINE, end="", flush=True)
    if not text or is_hallucination(text):
        context["prev_text"] = ""
        change = capture.adjust_silence(0)
        if change:
            session_logger.log_silence_change(*change)
        return

    duration = len(chunk) / SAMPLE_RATE

    # 文が完結していない & 蓄積に余裕がある → 次のチャンクと結合して再処理
    if not _is_sentence_end(text) and duration < MAX_PENDING_SECONDS:
        context["pending_audio"] = chunk
        print(f"\r{_DIM}  ... waiting{_RESET}", end="", flush=True)
        return

    # pending 表示をクリアして結果を出力
    print(_CLEAR_LINE, end="", flush=True)
    sentences = _split_sentences(text)
    context["prev_text"] = _build_context(sentences)
    change = capture.adjust_silence(len(sentences))
    if change:
        session_logger.log_silence_change(*change)
    # 要約コンテキストを翻訳に渡してIT文脈の訳し分けを改善
    session_context = summarizer.latest_summary
    print(f"\r{_YELLOW}⏳ Translating...{_RESET}", end="", flush=True)
    translated = _translate_sentences(
        sentences, translate_client, session_context
    )
    print(_CLEAR_LINE, end="", flush=True)
    _print_results(sentences, translated, session_logger)


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
    parser = argparse.ArgumentParser(
        description="macOS のシステム音声をリアルタイムで文字起こし・翻訳する CLI ツール",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="使用する AWS プロファイル名（デフォルト: default プロファイル）",
    )
    return parser.parse_args()


def _preload_whisper_model() -> None:
    """Whisper モデルを事前にロードする.

    初回の文字起こし時にモデルロードが走ると最初の発話を取りこぼすため、
    起動時にダミー音声で推論を実行してモデルをメモリに載せておく。
    """
    print("Loading Whisper model...", end="", flush=True)
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(
        dummy,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo-q4",
        language=LANGUAGE,
        without_timestamps=True,
    )
    print(f"{_CLEAR_LINE}Whisper model loaded.", flush=True)


def main() -> None:
    """メインループ. 音声キャプチャ→文字起こし→翻訳→表示を繰り返す."""
    args = _parse_args()
    _check_audio_output()
    _preload_whisper_model()
    translate_client = create_translate_client(profile=args.profile)
    session_logger = SessionLogger()
    summarizer = Summarizer(session_logger, profile=args.profile)
    summarizer.start()
    print(f"Log: {session_logger.path}")

    # スレッド間で共有するコンテキスト
    context: dict = {"prev_text": "", "pending_audio": None}
    # Whisper処理用のスレッドプール（1ワーカー = 順序を保証しつつメインループを非ブロック化）
    executor = ThreadPoolExecutor(max_workers=1)
    active_future: Future | None = None
    # ワーカー処理中に完了した発話チャンクを保持するバッファ
    queued_chunks: deque[np.ndarray] = deque()

    with AudioCapture(
        device_name=DEVICE_NAME,
        sample_rate=SAMPLE_RATE,
        sd_module=sd,
    ) as capture:
        try:
            while True:
                # 前回のワーカーが処理中なら音声キャプチャだけ続ける
                if active_future is not None and not active_future.done():
                    # 発話が完了したチャンクを保持（次のワーカーで処理する）
                    result = capture.get_audio_chunk()
                    if result is not None:
                        queued_chunks.append(result)
                    time.sleep(SLEEP_SECONDS)
                    continue

                # 前回のワーカーで例外が発生していたらログに記録
                if active_future is not None:
                    exc = active_future.exception()
                    if exc:
                        logger.exception("Processing failed", exc_info=exc)
                    active_future = None

                # ワーカー処理中に溜まったチャンクを優先的に処理
                if queued_chunks:
                    chunk = queued_chunks.popleft()
                else:
                    chunk = capture.get_audio_chunk()
                if chunk is None:
                    time.sleep(SLEEP_SECONDS)
                    continue

                # 前回の未完結音声があれば先頭に結合
                pending = context.get("pending_audio")
                if pending is not None:
                    chunk = np.concatenate([pending, chunk])
                    context["pending_audio"] = None

                # Whisperのinitial_promptを構築（話者名 + 要約ヒント + 直近テキスト）
                whisper_hint = summarizer.whisper_hint
                prev_text = context.get("prev_text", "")
                parts = []
                if KNOWN_SPEAKERS:
                    parts.append(", ".join(KNOWN_SPEAKERS))
                if whisper_hint:
                    parts.append(whisper_hint)
                if prev_text:
                    parts.append(prev_text)
                initial_prompt = " ".join(parts) if parts else None

                active_future = executor.submit(
                    _process_chunk,
                    chunk,
                    initial_prompt,
                    translate_client,
                    session_logger,
                    summarizer,
                    capture,
                    context,
                )
        except KeyboardInterrupt:
            summarizer.stop()
            executor.shutdown(wait=False)
            print(f"\nLog: {session_logger.path}")
