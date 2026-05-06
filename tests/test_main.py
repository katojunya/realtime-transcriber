"""mainモジュールのテスト.

メインループはAudioCapture → transcribe_audio → translate → printの
パイプラインを制御するオーケストレーション層。
モジュールを横断するデータフローのため、インテグレーションテストとして記述する。
"""

import argparse
from unittest.mock import MagicMock, patch

import numpy as np

from realtime_transcriber.main import (
    _build_context,
    _is_sentence_end,
    _split_sentences,
    main,
)


# --- ユーティリティ関数の単体テスト ---


class TestIsSentenceEnd:
    """_is_sentence_end関数のテスト."""

    def test_should_return_true_for_period(self) -> None:
        assert _is_sentence_end("Hello world.") is True

    def test_should_return_true_for_exclamation(self) -> None:
        assert _is_sentence_end("Hello world!") is True

    def test_should_return_true_for_question_mark(self) -> None:
        assert _is_sentence_end("Hello world?") is True

    def test_should_return_true_for_semicolon(self) -> None:
        assert _is_sentence_end("Hello world;") is True

    def test_should_return_false_for_ellipsis(self) -> None:
        # "..." は省略記号なので文末とみなさない
        assert _is_sentence_end("Hello world...") is False

    def test_should_return_false_for_empty_string(self) -> None:
        assert _is_sentence_end("") is False

    def test_should_return_false_for_incomplete_sentence(self) -> None:
        assert _is_sentence_end("Hello world") is False

    def test_should_strip_trailing_whitespace(self) -> None:
        assert _is_sentence_end("Hello world.  ") is True


class TestSplitSentences:
    """_split_sentences関数のテスト."""

    def test_should_split_on_period(self) -> None:
        result = _split_sentences("Hello. World.")
        assert result == ["Hello.", "World."]

    def test_should_not_split_on_abbreviation(self) -> None:
        # 略語のピリオドでは分割しない
        result = _split_sentences("Dr. Smith said hello. Then he left.")
        assert result == ["Dr. Smith said hello.", "Then he left."]

    def test_should_handle_single_sentence(self) -> None:
        result = _split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_should_handle_multiple_abbreviations(self) -> None:
        result = _split_sentences("Mr. Jones and Mrs. Brown arrived. They sat down.")
        assert result == ["Mr. Jones and Mrs. Brown arrived.", "They sat down."]


class TestBuildContext:
    """_build_context関数のテスト."""

    def test_should_return_last_sentences_within_limit(self) -> None:
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        result = _build_context(sentences)
        # 全体が200文字以内なら全文返る
        assert "First sentence." in result
        assert "Third sentence." in result

    def test_should_truncate_from_start_when_over_limit(self) -> None:
        # 長い文を作って200文字制限を超えさせる
        long = "A" * 150 + "."
        short = "Short."
        result = _build_context([long, short])
        # 末尾から取るので short は含まれる
        assert "Short." in result

    def test_should_return_empty_for_empty_list(self) -> None:
        assert _build_context([]) == ""


# --- メインループのインテグレーションテスト ---


def _make_mock_capture(side_effects: list) -> MagicMock:
    """テスト用のモックAudioCaptureを生成するヘルパー."""
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.get_audio_chunk.side_effect = side_effects
    mock.adjust_silence.return_value = None
    return mock


class _SyncExecutor:
    """テスト用の同期実行ThreadPoolExecutor代替."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args, **kwargs):
        """関数を同期的に実行してFutureを返す."""
        from concurrent.futures import Future

        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def map(self, fn, *iterables):
        """関数を同期的にmapして結果を返す."""
        return [fn(*args) for args in zip(*iterables)]

    def shutdown(self, **kwargs):
        pass


def _patch_main_deps(**overrides):
    """main()の外部依存をまとめてパッチするヘルパー.

    デフォルトで _check_audio_output, create_translate_client をモック化する。
    ThreadPoolExecutorを同期実行に差し替えてテストの確実性を担保する。
    overrides で追加のパッチを指定できる。
    """
    import contextlib

    patches = {
        "_parse_args": patch(
            "realtime_transcriber.main._parse_args",
            return_value=argparse.Namespace(profile=None),
        ),
        "_check_audio_output": patch("realtime_transcriber.main._check_audio_output"),
        "_preload_whisper_model": patch("realtime_transcriber.main._preload_whisper_model"),
        "create_translate_client": patch(
            "realtime_transcriber.main.create_translate_client",
            return_value=MagicMock(),
        ),
        "Summarizer": patch(
            "realtime_transcriber.main.Summarizer",
            return_value=MagicMock(
                whisper_hint="",
                latest_summary="",
            ),
        ),
        "ThreadPoolExecutor": patch(
            "realtime_transcriber.main.ThreadPoolExecutor",
            side_effect=lambda **kwargs: _SyncExecutor(**kwargs),
        ),
    }
    patches.update(overrides)

    @contextlib.contextmanager
    def combined():
        mocks = {}
        with contextlib.ExitStack() as stack:
            for name, p in patches.items():
                mocks[name] = stack.enter_context(p)
            yield mocks

    return combined()


class TestMainLoop:
    """main関数のインテグレーションテスト."""

    def test_should_translate_and_print_when_audio_available(self) -> None:
        # Given: 音声チャンクを1回返した後KeyboardInterruptで終了
        audio_chunk = np.random.rand(48000).astype(np.float32)
        mock_capture = _make_mock_capture([audio_chunk, KeyboardInterrupt])

        # When: mainを実行する
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio", return_value="Hello, world."):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=False):
                        with patch("realtime_transcriber.main.translate_text", return_value="こんにちは、世界。"):
                            with patch("builtins.print") as mock_print:
                                main()

        # Then: 原文（ANSI付き）と翻訳文が出力される
        printed_texts = [str(c.args[0]) if c.args else "" for c in mock_print.call_args_list]
        assert any("Hello, world." in t for t in printed_texts)
        assert any("こんにちは、世界。" in t for t in printed_texts)

    def test_should_not_print_when_transcription_is_empty(self) -> None:
        # Given: 空テキストを返す文字起こし
        audio_chunk = np.random.rand(48000).astype(np.float32)
        mock_capture = _make_mock_capture([audio_chunk, KeyboardInterrupt])

        # When: mainを実行する
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio", return_value=""):
                    with patch("realtime_transcriber.main.translate_text") as mock_translate:
                        with patch("builtins.print"):
                            main()

        # Then: 翻訳は呼ばれない
        mock_translate.assert_not_called()

    def test_should_skip_transcription_when_no_audio_chunk(self) -> None:
        # Given: get_audio_chunkがNoneを返し続ける
        mock_capture = _make_mock_capture([None, None, KeyboardInterrupt])
        mock_transcribe = MagicMock()

        # When: mainを実行する
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio", mock_transcribe):
                    with patch("time.sleep"):
                        main()

        # Then: 文字起こしは呼ばれない
        mock_transcribe.assert_not_called()

    def test_should_exit_gracefully_on_keyboard_interrupt(self) -> None:
        # Given: 即座にKeyboardInterruptが発生する
        mock_capture = _make_mock_capture([KeyboardInterrupt])

        # When/Then: 例外が発生せず正常終了する
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio"):
                    main()

        # Then: __exit__が呼ばれてリソースが解放される
        mock_capture.__exit__.assert_called_once()

    def test_should_sleep_when_no_audio_chunk_to_reduce_cpu(self) -> None:
        # Given: get_audio_chunkがNoneを返す
        mock_capture = _make_mock_capture([None, KeyboardInterrupt])

        # When: mainを実行する
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio"):
                    with patch("time.sleep") as mock_sleep:
                        main()

        # Then: CPU使用率抑制のためsleepが呼ばれる
        mock_sleep.assert_called_once()


class TestMainLoopHallucinationFilter:
    """メインループのハルシネーションフィルタのインテグレーションテスト."""

    def test_should_not_translate_hallucinated_text(self) -> None:
        # Given: Whisperがハルシネーションテキストを返す状況
        audio_chunk = np.full(48000, 0.5, dtype=np.float32)
        mock_capture = _make_mock_capture([audio_chunk, KeyboardInterrupt])

        # When: mainを実行する（ハルシネーションフィルタが検出）
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio", return_value="Thank you."):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=True):
                        with patch("realtime_transcriber.main.translate_text") as mock_translate:
                            with patch("builtins.print"):
                                main()

        # Then: ハルシネーションなので翻訳は呼ばれない
        mock_translate.assert_not_called()

    def test_should_translate_normal_text_after_hallucination_check(self) -> None:
        # Given: 正常なテキストを返すWhisper
        audio_chunk = np.full(48000, 0.5, dtype=np.float32)
        mock_capture = _make_mock_capture([audio_chunk, KeyboardInterrupt])

        # When: mainを実行する（ハルシネーションではない）
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio", return_value="Hello, world."):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=False):
                        with patch("realtime_transcriber.main.translate_text", return_value="こんにちは") as mock_translate:
                            with patch("builtins.print"):
                                main()

        # Then: 翻訳が呼ばれる
        mock_translate.assert_called_once()


class TestMainLoopPendingAudio:
    """未完結文のpending処理のインテグレーションテスト."""

    def test_should_defer_incomplete_sentence(self) -> None:
        # Given: 文末で終わらないテキスト → 次のチャンクで完結
        audio_chunk1 = np.random.rand(16000).astype(np.float32)  # 1秒
        audio_chunk2 = np.random.rand(16000).astype(np.float32)  # 1秒

        mock_capture = _make_mock_capture([audio_chunk1, audio_chunk2, KeyboardInterrupt])
        # 1回目: 未完結、2回目: 完結（結合された音声で再度文字起こし）
        transcribe_results = ["Hello world", "Hello world. How are you?"]

        # When: mainを実行する
        with _patch_main_deps():
            with patch("realtime_transcriber.main.AudioCapture", return_value=mock_capture):
                with patch("realtime_transcriber.main.transcribe_audio", side_effect=transcribe_results):
                    with patch("realtime_transcriber.main.is_hallucination", return_value=False):
                        with patch("realtime_transcriber.main.translate_text", return_value="翻訳結果"):
                            with patch("builtins.print"):
                                main()

        # Then: 2回目で翻訳されること（1回目はpendingでスキップ）
        # translate_textが呼ばれた = 完結した文が処理された
