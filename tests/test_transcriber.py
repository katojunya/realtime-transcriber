"""transcribe_audio関数とis_hallucination関数のテスト."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from realtime_transcriber.transcriber import (
    HALLUCINATION_PATTERNS,
    MODEL_REPO,
    _is_output_too_long,
    is_hallucination,
    transcribe_audio,
)


class TestIsHallucination:
    """is_hallucination関数のテスト."""

    def test_should_detect_thank_you(self) -> None:
        # Given: Whisperの典型的なハルシネーション "Thank you."

        # When: ハルシネーション判定する
        result = is_hallucination("Thank you.")

        # Then: ハルシネーションと判定される
        assert result is True

    def test_should_detect_thanks_for_watching(self) -> None:
        # Given: "Thanks for watching." パターン

        # When: ハルシネーション判定する
        result = is_hallucination("Thanks for watching.")

        # Then: ハルシネーションと判定される
        assert result is True

    def test_should_detect_bye(self) -> None:
        # Given: "Bye." パターン

        # When: ハルシネーション判定する
        result = is_hallucination("Bye.")

        # Then: ハルシネーションと判定される
        assert result is True

    def test_should_detect_the_end(self) -> None:
        # Given: "The end." パターン

        # When: ハルシネーション判定する
        result = is_hallucination("The end.")

        # Then: ハルシネーションと判定される
        assert result is True

    def test_should_detect_subtitles_by_amara(self) -> None:
        # Given: "Subtitles by the Amara.org community" パターン

        # When: ハルシネーション判定する
        result = is_hallucination("Subtitles by the Amara.org community")

        # Then: ハルシネーションと判定される
        assert result is True

    def test_should_be_case_insensitive(self) -> None:
        # Given: 大文字小文字が異なる "THANK YOU."

        # When: ハルシネーション判定する
        result = is_hallucination("THANK YOU.")

        # Then: 大文字小文字に関わらずハルシネーションと判定される
        assert result is True

    def test_should_strip_whitespace_before_matching(self) -> None:
        # Given: 前後に空白のある "  Thank you.  "

        # When: ハルシネーション判定する
        result = is_hallucination("  Thank you.  ")

        # Then: 空白を除去した上でハルシネーションと判定される
        assert result is True

    def test_should_not_detect_normal_speech(self) -> None:
        # Given: 通常の発話テキスト

        # When: ハルシネーション判定する
        result = is_hallucination("Hello, how are you doing today?")

        # Then: ハルシネーションではないと判定される
        assert result is False

    def test_should_not_detect_sentence_containing_thank_you(self) -> None:
        # Given: "Thank you" を含むが、それ自体が文全体ではないテキスト

        # When: ハルシネーション判定する
        result = is_hallucination("Thank you for your help with this project.")

        # Then: 完全一致ではないのでハルシネーションではない
        assert result is False

    def test_should_not_detect_empty_string(self) -> None:
        # Given: 空文字列

        # When: ハルシネーション判定する
        result = is_hallucination("")

        # Then: ハルシネーションではない
        assert result is False

    def test_should_not_detect_whitespace_only(self) -> None:
        # Given: 空白のみの文字列

        # When: ハルシネーション判定する
        result = is_hallucination("   ")

        # Then: ハルシネーションではない
        assert result is False


class TestHallucinationPatterns:
    """HALLUCINATION_PATTERNS定数のテスト."""

    def test_should_be_frozenset(self) -> None:
        # Given/When: 定数の型を確認する

        # Then: frozensetである（イミュータブル）
        assert isinstance(HALLUCINATION_PATTERNS, frozenset)

    def test_should_contain_common_hallucination_patterns(self) -> None:
        # Given: 既知のWhisperハルシネーションパターン
        expected_patterns = [
            "thank you.",
            "thanks for watching.",
            "bye.",
            "the end.",
        ]

        # When/Then: パターンが含まれている（小文字で格納されている前提）
        for pattern in expected_patterns:
            assert pattern in HALLUCINATION_PATTERNS, (
                f"'{pattern}' should be in HALLUCINATION_PATTERNS"
            )

    def test_should_store_patterns_in_lowercase(self) -> None:
        # Given/When: 全パターンを確認する

        # Then: 全て小文字で格納されている
        for pattern in HALLUCINATION_PATTERNS:
            assert pattern == pattern.lower(), (
                f"Pattern '{pattern}' should be lowercase"
            )

    def test_should_not_be_empty(self) -> None:
        # Given/When: パターン数を確認する

        # Then: 空ではない
        assert len(HALLUCINATION_PATTERNS) > 0


class TestTranscribeAudio:
    """transcribe_audio関数のテスト."""

    def test_should_return_transcribed_text(self) -> None:
        # Given: 音声データと正常にテキストを返すmlx_whisper
        audio = np.random.rand(16000 * 3).astype(np.float32)
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {
            "text": "Hello, world.",
        }

        # When: 文字起こしを実行する
        result = transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: 文字起こしテキストが返る
        assert result == "Hello, world."

    def test_should_pass_correct_model_repo(self) -> None:
        # Given: モック化されたmlx_whisper
        audio = np.random.rand(16000).astype(np.float32)
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {"text": "test"}

        # When: 文字起こしを実行する
        transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: 正しいモデルリポジトリが指定される
        call_kwargs = mock_mlx_whisper.transcribe.call_args
        assert call_kwargs.kwargs["path_or_hf_repo"] == MODEL_REPO

    def test_should_pass_language_in_decode_options(self) -> None:
        # Given: 英語指定
        audio = np.random.rand(16000).astype(np.float32)
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {"text": "test"}

        # When: 英語で文字起こしする
        transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: languageがdecode_optionsとして渡される
        call_kwargs = mock_mlx_whisper.transcribe.call_args
        assert call_kwargs.kwargs["language"] == "en"

    def test_should_pass_numpy_array_as_first_argument(self) -> None:
        # Given: numpy配列の音声データ
        audio = np.random.rand(16000).astype(np.float32)
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {"text": ""}

        # When: 文字起こしを実行する
        transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: numpy配列が第1引数として渡される
        call_args = mock_mlx_whisper.transcribe.call_args
        passed_audio = call_args.args[0] if call_args.args else call_args.kwargs.get("audio")
        assert isinstance(passed_audio, np.ndarray)

    def test_should_return_empty_string_when_no_speech_detected(self) -> None:
        # Given: 無音で空テキストを返すmlx_whisper
        audio = np.zeros(16000, dtype=np.float32)
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {"text": ""}

        # When: 無音の文字起こしを実行する
        result = transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: 空文字列が返る
        assert result == ""

    def test_should_propagate_transcription_error(self) -> None:
        # Given: エラーを投げるmlx_whisper
        audio = np.random.rand(16000).astype(np.float32)
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.side_effect = RuntimeError("Model load failed")

        # When/Then: エラーがそのまま伝播する
        with pytest.raises(RuntimeError, match="Model load failed"):
            transcribe_audio(
                audio=audio,
                language="en",
                mlx_whisper_module=mock_mlx_whisper,
            )

    def test_should_return_empty_when_output_too_long_for_audio(self) -> None:
        # Given: 1秒の音声に対して異常に長いテキストが出力された
        audio = np.random.rand(16000).astype(np.float32)  # 1秒
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {
            "text": "皆さん、こんにちは！今日はみんなに少し自己紹介してもらいたいと思います。" * 3
        }

        # When: 文字起こしを実行する
        result = transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: 異常な出力として空文字が返る
        assert result == ""

    def test_should_return_text_when_output_length_is_normal(self) -> None:
        # Given: 3秒の音声に対して妥当な長さのテキスト
        audio = np.random.rand(16000 * 3).astype(np.float32)  # 3秒
        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {
            "text": "Hello everyone, welcome to today's presentation."
        }

        # When: 文字起こしを実行する
        result = transcribe_audio(
            audio=audio,
            language="en",
            mlx_whisper_module=mock_mlx_whisper,
        )

        # Then: 正常にテキストが返る
        assert result == "Hello everyone, welcome to today's presentation."


class TestIsOutputTooLong:
    """_is_output_too_long関数のテスト."""

    def test_should_detect_too_long_output(self) -> None:
        # Given: 1秒の音声に対して100文字の出力（30文字/秒を超過）
        audio = np.random.rand(16000).astype(np.float32)
        text = "a" * 100

        # When/Then: 異常と判定される
        assert _is_output_too_long(text, audio) is True

    def test_should_allow_normal_output(self) -> None:
        # Given: 3秒の音声に対して50文字の出力（約17文字/秒）
        audio = np.random.rand(16000 * 3).astype(np.float32)
        text = "Hello everyone welcome to the presentation."

        # When/Then: 正常と判定される
        assert _is_output_too_long(text, audio) is False

    def test_should_return_false_when_text_is_empty(self) -> None:
        audio = np.random.rand(16000).astype(np.float32)
        assert _is_output_too_long("", audio) is False
