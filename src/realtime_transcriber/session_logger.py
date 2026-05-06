"""セッションログモジュール.

起動ごとにログファイルを作成し、文字起こし・翻訳結果をタイムスタンプ付きで記録する。
ログは logs/ ディレクトリに保存される。
要約機能のために、前回要約以降のテキストを取得する機能も持つ。
"""

import threading
from datetime import datetime
from pathlib import Path

# ログ出力先ディレクトリ（プロジェクトルート/logs/）
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class SessionLogger:
    """1セッション（1回の起動）に対応するログファイルを管理する."""

    def __init__(self) -> None:
        _LOGS_DIR.mkdir(exist_ok=True)
        self._start_time = datetime.now()
        # ミリ秒まで含めてファイル名の衝突を防ぐ
        filename = self._start_time.strftime("%Y-%m-%d_%H%M%S_%f.log")
        self._path = _LOGS_DIR / filename
        # 万が一同名ファイルが存在しても追記ではなく新規作成する
        self._path.write_text(
            f"# Session started at {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            encoding="utf-8",
        )
        # 要約用: 前回の要約以降に蓄積された翻訳テキスト（スレッドセーフ）
        self._recent_entries: list[str] = []
        self._recent_lock = threading.Lock()

    @property
    def path(self) -> Path:
        """ログファイルのパスを返す."""
        return self._path

    def elapsed(self) -> str:
        """セッション開始からの経過時間を [MM:SS] 形式で返す."""
        delta = datetime.now() - self._start_time
        total_seconds = int(delta.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        return f"[{minutes:02d}:{seconds:02d}]"

    def log(self, sentence: str, translated: str) -> None:
        """原文と翻訳文を1エントリとして記録する."""
        ts = self.elapsed()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} {sentence}\n")
            f.write(f"{ts} {translated}\n\n")
        # 要約用に翻訳テキストを蓄積
        with self._recent_lock:
            self._recent_entries.append(f"{ts} {translated}")

    def flush_recent(self) -> list[str]:
        """前回の要約以降に蓄積されたテキストを返し、バッファをクリアする."""
        with self._recent_lock:
            entries = self._recent_entries
            self._recent_entries = []
            return entries

    def log_summary(self, summary: str) -> None:
        """要約をログファイルに記録する."""
        ts = self.elapsed()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(f"--- {ts} 要約 ---\n")
            f.write(f"{summary}\n")
            f.write("---\n\n")

    def log_silence_change(self, prev_ms: int, new_ms: int) -> None:
        """無音閾値の変更をログファイルに記録する."""
        ts = self.elapsed()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} [silence] {prev_ms}ms → {new_ms}ms\n\n")

    def log_whisper_hint(self, prompt_en: str) -> None:
        """Whisper用英語ヒントをログファイルに記録する."""
        ts = self.elapsed()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(f"--- {ts} whisper_hint ---\n")
            f.write(f"{prompt_en}\n")
            f.write("---\n\n")
