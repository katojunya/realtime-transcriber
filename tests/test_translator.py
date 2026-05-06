"""translator モジュールのテスト.

Ollama / Bedrock / AWS Translate の各バックエンドをテストする。
"""

from unittest.mock import MagicMock

import pytest

from realtime_transcriber import translator
from realtime_transcriber.translator import translate_text


class TestTranslateTextOllama:
    """Ollamaバックエンドのテスト."""

    def setup_method(self) -> None:
        self._original_backend = translator.TRANSLATION_BACKEND
        translator.TRANSLATION_BACKEND = "ollama"

    def teardown_method(self) -> None:
        translator.TRANSLATION_BACKEND = self._original_backend

    def _make_client(self, translated: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": translated}
        }
        return mock_client

    def test_should_return_translated_text(self) -> None:
        # Given: 正常に翻訳を返すOllamaクライアント
        mock_client = self._make_client("こんにちは、世界。")

        # When: 翻訳を実行する
        result = translate_text("Hello, world.", "en", "ja", mock_client)

        # Then: 日本語翻訳テキストが返る
        assert result == "こんにちは、世界。"

    def test_should_call_chat_api_with_configured_model(self) -> None:
        # Given: モック化されたOllamaクライアント
        mock_client = self._make_client("テスト")

        # When: 翻訳を実行する
        translate_text("test", "en", "ja", mock_client)

        # Then: Ollama Chat APIが正しいモデル名・温度で呼ばれる
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == translator.OLLAMA_TRANSLATE_MODEL
        assert call_kwargs["options"]["temperature"] == 0.1
        # systemプロンプトとユーザーテキストが渡される
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test"

    def test_should_include_session_context_in_system_prompt(self) -> None:
        # Given: モック化されたOllamaクライアント
        mock_client = self._make_client("テスト")

        # When: session_contextを渡して翻訳を実行する
        translate_text("test", "en", "ja", mock_client, session_context="AWS re:Invent")

        # Then: systemプロンプトにsession_contextが含まれる
        messages = mock_client.chat.call_args[1]["messages"]
        assert "AWS re:Invent" in messages[0]["content"]

    def test_should_strip_whitespace_from_response(self) -> None:
        # Given: 前後に空白を含むレスポンス
        mock_client = self._make_client("  翻訳結果  \n")

        # When: 翻訳を実行する
        result = translate_text("result", "en", "ja", mock_client)

        # Then: 前後の空白が除去される
        assert result == "翻訳結果"

    def test_should_propagate_api_error(self) -> None:
        # Given: エラーを返すOllamaクライアント
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("ConnectionError")

        # When/Then: エラーがそのまま伝播する
        with pytest.raises(Exception, match="ConnectionError"):
            translate_text("Hello", "en", "ja", mock_client)


class TestTranslateTextBedrock:
    """Bedrockバックエンドのテスト."""

    def setup_method(self) -> None:
        self._original_backend = translator.TRANSLATION_BACKEND
        translator.TRANSLATION_BACKEND = "bedrock"

    def teardown_method(self) -> None:
        translator.TRANSLATION_BACKEND = self._original_backend

    def _make_client(self, translated: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": translated}]}}
        }
        return mock_client

    def test_should_return_translated_text(self) -> None:
        # Given: 正常に翻訳を返すBedrockクライアント
        mock_client = self._make_client("こんにちは、世界。")

        # When: 翻訳を実行する
        result = translate_text("Hello, world.", "en", "ja", mock_client)

        # Then: 日本語翻訳テキストが返る
        assert result == "こんにちは、世界。"

    def test_should_call_converse_api(self) -> None:
        # Given: モック化されたBedrockクライアント
        mock_client = self._make_client("テスト")

        # When: 翻訳を実行する
        translate_text("test", "en", "ja", mock_client)

        # Then: Bedrock Converse APIが呼ばれる
        mock_client.converse.assert_called_once()
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.1

    def test_should_strip_whitespace_from_response(self) -> None:
        # Given: 前後に空白を含むレスポンス
        mock_client = self._make_client("  翻訳結果  \n")

        # When: 翻訳を実行する
        result = translate_text("result", "en", "ja", mock_client)

        # Then: 前後の空白が除去される
        assert result == "翻訳結果"

    def test_should_propagate_api_error(self) -> None:
        # Given: エラーを返すBedrockクライアント
        mock_client = MagicMock()
        mock_client.converse.side_effect = Exception("ServiceUnavailable")

        # When/Then: エラーがそのまま伝播する
        with pytest.raises(Exception, match="ServiceUnavailable"):
            translate_text("Hello", "en", "ja", mock_client)


class TestTranslateTextAwsTranslate:
    """AWS Translateバックエンドのテスト."""

    def setup_method(self) -> None:
        self._original_backend = translator.TRANSLATION_BACKEND
        translator.TRANSLATION_BACKEND = "aws_translate"

    def teardown_method(self) -> None:
        translator.TRANSLATION_BACKEND = self._original_backend

    def _make_client(self, translated: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.translate_text.return_value = {
            "TranslatedText": translated,
            "SourceLanguageCode": "en",
            "TargetLanguageCode": "ja",
        }
        return mock_client

    def test_should_return_translated_text(self) -> None:
        # Given: 正常に翻訳を返すAWS Translateクライアント
        mock_client = self._make_client("こんにちは、世界。")

        # When: 翻訳を実行する
        result = translate_text("Hello, world.", "en", "ja", mock_client)

        # Then: 日本語翻訳テキストが返る
        assert result == "こんにちは、世界。"

    def test_should_call_translate_with_correct_parameters(self) -> None:
        # Given: モック化されたAWS Translateクライアント
        mock_client = self._make_client("テスト")

        # When: 翻訳を実行する
        translate_text("test", "en", "ja", mock_client)

        # Then: 正しいパラメータでAPIが呼ばれる
        mock_client.translate_text.assert_called_once_with(
            Text="test",
            SourceLanguageCode="en",
            TargetLanguageCode="ja",
        )

    def test_should_propagate_api_error(self) -> None:
        # Given: エラーを返すAWS Translateクライアント
        mock_client = MagicMock()
        mock_client.translate_text.side_effect = Exception("ServiceUnavailable")

        # When/Then: エラーがそのまま伝播する
        with pytest.raises(Exception, match="ServiceUnavailable"):
            translate_text("Hello", "en", "ja", mock_client)
