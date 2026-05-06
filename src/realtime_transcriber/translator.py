"""翻訳モジュール.

Ollama（ローカルLLM） / Amazon Bedrock / AWS Translate による英日翻訳を担当する。
TRANSLATION_BACKEND で切り替え可能。デフォルトは "ollama"。

各設定は同名の環境変数が定義されていればそれを優先する。
- TRANSLATION_BACKEND: "ollama" / "bedrock" / "aws_translate"
- OLLAMA_HOST: Ollamaサーバーのエンドポイント
- OLLAMA_TRANSLATE_MODEL: Ollamaで使用する翻訳モデル
- BEDROCK_MODEL_ID: Bedrockで使用するモデルID
"""

import logging
import os

logger = logging.getLogger(__name__)

# --- 翻訳設定 ---
# 翻訳バックエンド: "ollama" / "bedrock" / "aws_translate"
TRANSLATION_BACKEND = os.environ.get("TRANSLATION_BACKEND", "ollama")

# --- Ollama設定 ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# 翻訳に使うOllamaモデル。低レイテンシ重視で小〜中サイズが望ましい。
# 例: "gemma4:e4b"（既定）, "gemma3:4b", "gpt-oss:20b", "gemma4:31b"
OLLAMA_TRANSLATE_MODEL = os.environ.get("OLLAMA_TRANSLATE_MODEL", "gemma4:e4b")
# Ollamaサーバーがモデルをメモリに保持する時間。
# 翻訳のたびにロードされる事故を防ぐため、セッション中は常駐させる。
# "30m" / "1h" / "-1"（無期限） などを指定可能。
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# --- Bedrock設定 ---
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
# 使用するBedrockモデル（クロスリージョン推論プロファイル）
# - Amazon Nova 2 Lite: "us.amazon.nova-2-lite-v1:0"（高品質・高速・低コスト）
# - Amazon Nova Pro:    "us.amazon.nova-pro-v1:0"
# - Claude Haiku 4.5:   "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")

# --- AWS Translate設定 ---
AWS_TRANSLATE_REGION = os.environ.get("AWS_TRANSLATE_REGION", "ap-northeast-1")


def create_translate_client(profile: str | None = None) -> object:
    """翻訳クライアントを生成する.

    TRANSLATION_BACKEND に応じて Ollama / Bedrock / AWS Translate のクライアントを返す。

    Args:
        profile: AWS プロファイル名（Bedrock / AWS Translate 利用時のみ参照）。
    """
    if TRANSLATION_BACKEND == "ollama":
        from ollama import Client

        return Client(host=OLLAMA_HOST)

    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=profile)
    if TRANSLATION_BACKEND == "bedrock":
        config = Config(retries={"max_attempts": 5, "mode": "adaptive"})
        return session.client(
            "bedrock-runtime", region_name=BEDROCK_REGION, config=config
        )
    return session.client("translate", region_name=AWS_TRANSLATE_REGION)


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    client: object,
    session_context: str = "",
) -> str:
    """テキストを翻訳する.

    TRANSLATION_BACKEND に応じて Ollama / Bedrock / AWS Translate を使用する。

    Args:
        text: 翻訳元テキスト
        source_lang: ソース言語コード（例: "en"）
        target_lang: ターゲット言語コード（例: "ja"）
        client: 翻訳クライアント
        session_context: セッションの要約（LLM翻訳の文脈として使用）

    Returns:
        翻訳されたテキスト
    """
    if TRANSLATION_BACKEND == "ollama":
        return _translate_with_ollama(
            text, source_lang, target_lang, client, session_context
        )
    if TRANSLATION_BACKEND == "bedrock":
        return _translate_with_bedrock(
            text, source_lang, target_lang, client, session_context
        )
    return _translate_with_aws_translate(text, source_lang, target_lang, client)


def _build_translation_system_prompt(
    source_lang: str, target_lang: str, session_context: str
) -> str:
    """翻訳用のsystemプロンプトを組み立てる（OllamaとBedrockで共通）."""
    prompt = (
        f"Translate the following {source_lang} text to natural {target_lang}. "
        "Translate so that it is easy for Japanese speakers to understand. "
        "For technical terms that are commonly used in English (e.g. AWS, API), "
        "keep them in English. "
        "Output ONLY the translated text. "
        "Do not add explanations, context, or commentary."
    )
    if session_context:
        prompt += f"\n\nSession context for reference: {session_context}"
    return prompt


def _translate_with_ollama(
    text: str,
    source_lang: str,
    target_lang: str,
    client: object,
    session_context: str = "",
) -> str:
    """Ollamaでテキストを翻訳する."""
    system_prompt = _build_translation_system_prompt(
        source_lang, target_lang, session_context
    )
    response = client.chat(
        model=OLLAMA_TRANSLATE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        # 思考型モデル（qwen3系等）の thinking を無効化してレイテンシを抑える
        think=False,
        keep_alive=OLLAMA_KEEP_ALIVE,
        options={"temperature": 0.1, "num_predict": 256},
    )
    return response["message"]["content"].strip()


def _translate_with_bedrock(
    text: str,
    source_lang: str,
    target_lang: str,
    client: object,
    session_context: str = "",
) -> str:
    """Bedrockでテキストを翻訳する.

    systemフィールドに翻訳指示を分離し、userロールには翻訳対象テキストのみを渡す。
    """
    system_prompt = _build_translation_system_prompt(
        source_lang, target_lang, session_context
    )
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": text}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def _translate_with_aws_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    client: object,
) -> str:
    """AWS Translateでテキストを翻訳する."""
    response = client.translate_text(
        Text=text,
        SourceLanguageCode=source_lang,
        TargetLanguageCode=target_lang,
    )
    return response["TranslatedText"]
