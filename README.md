# realtime-transcriber のローカルLLM対応

オリジナルの
[realtime-transcriber](https://qiita.com/s_moriyama/items/af1110bb8566136adb23)は、@s_moriyama (Satoshi Moriyama) さんが作成した、英語音声をリアルタイム文字おこし翻訳をしてくれるCLIツールです。qiita に解説と、github にコードを公開してくれています。

オリジナルはリアルタイム翻訳と1分ごとの要約機能はAWSに依存しているため、本改造ではローカルLLM (Ollama)でモデルを使う機能を付加しました。Claude Code を使って実装しています。またローカルLLMはデフォルトでモデルはgemma4:e4bを使っています。

# realtime-transcriber

macOS のシステム音声をリアルタイムで文字起こし・翻訳する CLI ツールです。

このツールはシステム音声を直接キャプチャするため、YouTube Live・ウェビナー・Zoom・Teams など任意のアプリの英語音声を、プラットフォームや配信者の設定に関係なくリアルタイムで日本語に翻訳できます。

BlackHole 2ch 経由でキャプチャした音声を MLX-Whisper（`mlx-community/whisper-large-v3-turbo-q4`）で文字起こしし、Ollama 経由のローカル LLM（デフォルト: `gemma4:e4b`）で日本語に翻訳して表示します。60秒ごとにローカル LLM（デフォルト: `gemma4:e4b`）で内容を要約し、Whisper の認識精度向上にも活用します。Amazon Bedrock へのバックエンド切替も可能です。

## アーキテクチャ

```
システム音声 → BlackHole 2ch → sounddevice
  → VAD（Silero VAD）で発話区間を検出
  → MLX-Whisper で文字起こし（要約から生成した英語ヒントで精度向上）
  → ハルシネーション除去（定型フレーズ・繰り返しパターン検出）
  → Ollama（ローカルLLM）で英→日翻訳（複数文は並列処理、バックエンド切替可能）
  → ターミナルに表示 + ログファイルに記録

  [60秒ごと]
  → Ollama（ローカルLLM）で日本語要約 + 英語キーワード要約を生成
  → 英語キーワード要約を Whisper の initial_prompt に反映
```

## 必要なディスク容量

- MLX-Whisper モデル（`whisper-large-v3-turbo-q4`）: 約 400 MB（初回起動時に `~/.cache/huggingface/hub/` へ自動ダウンロード）
- Python 依存パッケージ: 数百 MB

## 前提条件

- macOS（Apple Silicon）
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/) （パッケージ管理）
- [BlackHole 2ch](https://existential.audio/blackhole/) （仮想オーディオデバイス）
- macOS の「Audio MIDI 設定」で Multi-Output Device（複数出力装置） を作成済み（後述）
- [Ollama](https://ollama.com/) がインストールされ、サーバーが起動済み
  - 翻訳・要約用モデル（既定: `gemma4:e4b`）を `ollama pull` 済み

AWS Bedrock を使う場合のみ別途 AWS CLI による SSO ログインとモデルアクセスの有効化が必要です（後述「翻訳バックエンドの切り替え」参照）。

## セットアップ

### 1. Multi-Output Device（複数出力装置） の作成

BlackHole 2ch をインストールしたら、macOS の「Audio MIDI 設定」で Multi-Output Device（複数出力装置） を作成します。

1. 「Audio MIDI 設定」を開く（Spotlight で「Audio MIDI」と検索）
2. 左下の「+」ボタン →「複数出力装置を作成」
3. 「BlackHole 2ch」と通常のスピーカー（例: MacBook Pro のスピーカー）の両方にチェックを入れる

これにより、音声がスピーカーから聞こえると同時に BlackHole 経由でアプリにもキャプチャされます。

### 2. 依存パッケージのインストール

```bash
uv sync
```

依存パッケージのバージョンは `uv.lock` で固定されているため、再現性のある環境が構築されます。

### 3. Ollama モデルの取得

既定では翻訳・要約ともに `gemma4:e4b` を使用します。

```bash
ollama pull gemma4:e4b
```

別のモデルを使いたい場合は同様に `ollama pull <model>` でダウンロードし、環境変数 `OLLAMA_TRANSLATE_MODEL` / `OLLAMA_SUMMARY_MODEL` で指定します（後述）。

Ollama サーバーが `http://localhost:11434` 以外で動いている場合は `OLLAMA_HOST` を設定してください。

## 使い方

```bash
uv run realtime-transcriber
```

起動すると音声出力先の確認が行われます。Multi-Output Device（複数出力装置） に切り替えた後、Enter を押すと文字起こしが開始されます。

ログファイルは `logs/` ディレクトリにセッションごとに生成されます。

終了は `Ctrl+C` です。

### 別モデルの利用

```bash
OLLAMA_TRANSLATE_MODEL=gemma4:31b OLLAMA_SUMMARY_MODEL=qwen3.6:latest uv run realtime-transcriber
```

### Bedrock / AWS Translate を使う場合

```bash
TRANSLATION_BACKEND=bedrock SUMMARIZER_BACKEND=bedrock uv run realtime-transcriber --profile your-profile-name
```

## 出力例

```
● Recording... 5s ▼
  [00:21] We're driven by the idea that the products and services we create
          should help people unleash their creativity and potential.
  [00:21] 私たちは、私たちが作る製品やサービスが人々の創造性と可能性を
          引き出す手助けをするべきだという考えに導かれています。

  [00:27] We build some of the largest internet services on the planet
          and many of them run on AWS.
  [00:27] 私たちは世界最大級のインターネットサービスの一部を構築しており、
          それらの多くはAWS上で稼働しています。

--- 要約 ---
ペイアム・ウラシディ氏が登壇し、Appleのクラウドインフラストラクチャ戦略に
ついて説明しました。同氏のチームはApp Store、Apple Music、Apple TV、
Podcastsなど、数十億人が利用する主要サービスを開発・運営しており、これらは
AWSと自社データセンターの両方で稼働しています。
---
```

## 処理の仕組み

### 音声キャプチャと VAD（発話区間検出）

BlackHole 2ch 経由でシステム音声をステレオで取得し、モノラルに変換して Silero VAD に渡します。VAD が発話区間を検出すると音声バッファへの蓄積を開始し、以下の条件で区切ります。

- 無音が一定時間続いた時点で発話終了と判定（初期値 500ms、前回の翻訳結果の文数に応じて 200ms〜800ms の範囲で動的に調整）
- 発話が 30 秒に達したら強制的に区切り
- 1 秒未満の発話はノイズとして無視

### 文の結合と分割

Whisper の出力が文末（`.` `!` `?` `;`）で終わっていない場合、音声を次のチャンクと結合して再処理します。これにより文の途中で切られることを防ぎます。ただし 15 秒以上蓄積しても文が完結しない場合は、そのまま翻訳に回します（最大で 30 + 15 = 45 秒分の音声がつながる可能性があります）。

文が完結したら、略語（Mr. Dr. e.g. など）のピリオドで誤分割しないよう保護しつつ、文単位に分割して並列翻訳します。

### Whisper のコンテキスト引き継ぎ

Whisper の `initial_prompt` に以下を渡して認識精度を向上させています。

- 直近の文字起こし結果（最大 200 文字、文単位で切り出し）
- 60 秒ごとの要約から生成された英語キーワード（最大 400 文字）

これにより、セッション固有の専門用語や固有名詞が文字起こしに反映されやすくなります。

### ハルシネーション除去

Whisper が無音や短い音声に対して出力する定型フレーズ（"Thank you." "Bye." など）をフィルタリングします。加えて以下のパターンも検出します。

- 同じ単語の 5 回以上の連続（例: "too too too too too"）
- 同じ文字の 10 回以上の連続（例: "llllllllll"）
- 同じ 2〜6 文字パターンの 5 回以上の繰り返し（例: "結論の結論の結論の..."）
- 英数字がほぼ含まれないテキスト

繰り返しが検出された場合、除去後のテキストが元の 50% 未満ならハルシネーションとして破棄し、50% 以上残る場合は繰り返し部分のみ除去して翻訳に回します。

### 定期要約（60 秒ごと）

バックグラウンドのデーモンスレッドが 60 秒ごとにローカル LLM（既定: `gemma4:e4b`）を呼び出し、前回の要約と直近の翻訳テキストから更新された要約を生成します。1 回の API 呼び出しで以下の 2 つを同時に生成します。

- 日本語要約（ターミナル表示・ログ記録用）: 英語圏特有の表現や略語を補足した自然な文章
- 英語キーワード要約（Whisper の `initial_prompt` 用）: セッションのトピック、専門用語、話者名などを 400 文字以内で記述

### ステータス表示

ターミナルにはリアルタイムで処理状態が表示されます。

- `● Recording... Ns` — VAD が発話を検出し、音声を蓄積中（N は秒数）。無音閾値が変更された場合は ▼（短縮）/ ▲（延長）が表示される
- `⏳ Transcribing...` — Whisper が音声を文字起こし中
- `⏳ Translating...` — 翻訳中
- `... waiting` — 文が未完結のため、次のチャンクを待機中

## 翻訳バックエンドの切り替え

`TRANSLATION_BACKEND` / `SUMMARIZER_BACKEND` の環境変数（または `translator.py` / `summarizer.py` の同名定数）で切り替えられます。環境変数が定義されていれば常に環境変数が優先されます。

| バックエンド | 値 | 説明 |
|------------|-----|------|
| Ollama（デフォルト） | `ollama` | ローカルLLMで翻訳・要約。完全オフライン・無料 |
| Amazon Bedrock | `bedrock` | クラウドLLMで翻訳・要約。高品質、AWSアカウント必要 |
| AWS Translate | `aws_translate` | 機械翻訳サービス（翻訳のみ。要約には使えない） |

### Ollama 使用時の設定

```bash
# 翻訳モデル（低レイテンシ重視で小〜中サイズが望ましい）
OLLAMA_TRANSLATE_MODEL=gemma4:e4b    # 既定

# 要約モデル（日本語生成とJSON出力が安定しているもの）
OLLAMA_SUMMARY_MODEL=gemma4:e4b      # 既定

# Ollamaサーバーのエンドポイント
OLLAMA_HOST=http://localhost:11434   # 既定

# モデルをメモリに保持する時間
OLLAMA_KEEP_ALIVE=30m                # 既定
```

利用候補（事前に `ollama pull` が必要）:

| モデル | サイズ | 用途 | 備考 |
|--------|--------|------|------|
| `gemma4:e4b`（翻訳・要約の既定） | 9.6GB | 翻訳・要約 | 軽量・高速・1秒未満の翻訳が可能 |
| `gpt-oss:20b` | 13GB | 翻訳・要約 | バランスが良く、JSON出力も安定 |
| `gemma4:31b` | 19GB | 翻訳・要約 | より高品質だが遅い |
| `qwen3.6:latest` | 23GB | 要約 | 多言語に強いが thinking モデルなので `think=False` で運用 |
| `gpt-oss:120b` | 65GB | 要約 | 高品質だが重い。VRAMに余裕があれば |

### Bedrock 使用時の設定

```bash
TRANSLATION_BACKEND=bedrock
SUMMARIZER_BACKEND=bedrock

# 翻訳モデル
BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0           # 既定
# 例: us.amazon.nova-pro-v1:0
# 例: us.anthropic.claude-haiku-4-5-20251001-v1:0

# 要約モデル
BEDROCK_SUMMARY_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0  # 既定

BEDROCK_REGION=us-east-1   # 既定
```

Bedrock のモデルアクセスは us-east-1 リージョンで有効化してください（クロスリージョン推論プロファイルを使用するため）。AWS CLI の `aws sso login` または `--profile` オプションで認証情報を渡します。

### 翻訳品質の比較（Bedrockバックエンド）

同じ英語セッション（Apple の re:Invent 登壇）に対して実際に出力された翻訳の比較です。

| 英語原文 | AWS Translate | Nova Lite | Nova Pro | Haiku 4.5 |
|---------|--------------|-----------|----------|-----------|
| power everything from Apple Silicon | **電力を供給** | **動かして** | **動かすために使用** | **動かして** |
| we're a Java shop | **Javaショップ** | **Javaショップ** | **Javaを主に使用** | **Javaを使う企業** |
| can raise eyebrows | 眉をひそめる | 眉をひそめさせる（誤解釈あり） | **注目を集める可能性** | 眉をひそめられる |
| a win for the planet | **地球**にとってもメリット | **地球**にとってより小さい… | お客様と**地球**にとってwin-win | **プラネット**にとっても… |
| homomorphic encryption | ― | 同型暗号化 | **ホモモルフィック暗号化**（補足付き） | 準同型暗号化 |

LLM翻訳ではプロンプトで「日本語話者にわかりやすく、技術用語は英語のまま」と指示しているため、AWS, API, Swift, Graviton などは英語のまま出力します。

### 思考型モデル（Qwen3 系等）について

Qwen3 のような thinking 機能を持つモデルでは、Ollama の chat API に `think=False` を渡して thinking を無効化しています（[translator.py](src/realtime_transcriber/translator.py), [summarizer.py](src/realtime_transcriber/summarizer.py) で設定）。有効のままだと `num_predict` 上限を thinking で使い切って `content` が空になる、応答レイテンシが大幅に増加する、などの問題が発生します。

## 主要モジュール

| ファイル | 役割 |
|---------|------|
| `main.py` | CLI エントリポイント。パイプライン全体の制御 |
| `audio.py` | 音声キャプチャと VAD による発話区間検出 |
| `transcriber.py` | MLX-Whisper による文字起こしとハルシネーション除去 |
| `translator.py` | Ollama / Bedrock / AWS Translate による翻訳（切り替え可能） |
| `summarizer.py` | Ollama / Bedrock による定期要約（日本語要約 + Whisper用英語ヒント） |
| `session_logger.py` | セッションごとのログファイル管理 |

## 料金の目安

### Ollama（デフォルト）

すべてローカル実行のため **無料**。Ollama サーバーが動作する Mac の電力消費のみ。

### Bedrock 翻訳

- Nova 2 Lite: 翻訳 ~$0.09/h、要約（Haiku 4.5）~$0.07/h、合計 約 **$0.16 / 時間**
- Nova Pro: 翻訳 ~$0.13/h、合計 約 **$0.20 / 時間**
- Haiku 4.5: 翻訳 ~$0.20/h、合計 約 **$0.30 / 時間**

### AWS Translate 翻訳 + Bedrock 要約

- $15.00 / 100万文字（翻訳）+ Haiku 4.5 要約（~$0.07/h）
- 合計目安: 約 **$0.15〜$0.30 / 時間**

## テスト

```bash
uv run pytest
```

## 既知の制限事項

- 英語→日本語のみ対応（他の言語ペアは未検証）
- macOS（Apple Silicon）専用。Linux / Windows では動作しない
- 仮想オーディオデバイスは BlackHole 2ch のみ検証済み（Soundflower 等は未検証）
- Whisper の認識精度はスピーカーの発音、音質、背景ノイズに依存する
- 話者名の認識は不安定（セッションごとに `KNOWN_SPEAKERS` を設定すると改善）

## トラブルシューティング

### 「Device 'BlackHole 2ch' not found」と表示される

BlackHole 2ch がインストールされていないか、認識されていません。[BlackHole](https://existential.audio/blackhole/) をインストールし、「Audio MIDI 設定」で表示されることを確認してください。

### 音声が取れない / Recording... が表示されない

以下を確認してください。

1. macOS のシステム出力が Multi-Output Device（複数出力装置） に切り替わっているか（「システム設定 → サウンド → 出力」で確認）
2. ターミナルアプリにマイクのアクセス許可があるか（「システム設定 → プライバシーとセキュリティ → マイク」で Terminal.app や iTerm2 等を許可）

BlackHole はシステム音声をキャプチャしますが、macOS はこれを「マイク入力」として扱うため、アプリにマイク許可が必要です。

### Translation failed が繰り返し表示される（Ollama）

以下を確認してください。

1. `ollama serve` が起動しているか（`curl http://localhost:11434/api/version` で確認）
2. 指定したモデルが pull 済みか（`ollama list` で確認）
3. Ollama のメモリ不足でモデルがロードできていないか（小さいモデルに切り替える）

### Bedrock で AccessDeniedException が発生する

`TRANSLATION_BACKEND=bedrock` で運用している場合は以下を確認してください。

1. `--profile` オプションまたは `AWS_PROFILE` 環境変数が正しく設定されているか
2. `aws sso login` でログイン済みか
3. us-east-1 リージョンで Bedrock モデルアクセスが有効か（[Bedrock コンソール](https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess) で確認）

## 技術的な補足

### silero-vad-lite について

VAD（Voice Activity Detection: 音声区間検出）には [silero-vad-lite](https://github.com/snakers4/silero-vad) を使用しています。これは Silero VAD の軽量版で、PyTorch に依存せず ONNX Runtime で動作するため、インストールサイズが小さく起動も高速です。精度は通常版の Silero VAD と同等です。

## ライセンス

MIT License
