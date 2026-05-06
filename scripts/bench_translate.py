"""翻訳バックエンドのレイテンシ計測ベンチマーク.

Ollamaの翻訳モデルが目標レイテンシ（短文 ~1s, 中文 ~2s）に収まるか検証する。

使い方:
    # デフォルトモデル（OLLAMA_TRANSLATE_MODEL もしくは translator.py の既定値）で計測
    uv run python scripts/bench_translate.py

    # モデルを指定して計測
    uv run python scripts/bench_translate.py --model gemma3:4b
    uv run python scripts/bench_translate.py --model gemma4:e4b

    # 複数モデルを順に比較
    uv run python scripts/bench_translate.py --models gemma3:4b gemma4:e4b gpt-oss:20b

注意:
    初回呼び出しはモデルのロード時間が含まれるため計測対象から除外する（warmup）。
    keep_alive により2回目以降はメモリに常駐したまま推論される。
"""

from __future__ import annotations

import argparse
import statistics
import time

from realtime_transcriber import translator


# 実セッションを想定した英文サンプル（短文〜長文を混ぜる）
SAMPLES: list[str] = [
    # 短文（〜10語）: ~0.5-1s 目標
    "Welcome to AWS re:Invent.",
    "Let's get started.",
    "Thank you for joining us today.",
    # 中文（10〜25語）: ~1-2s 目標
    "Today we'll talk about Amazon Bedrock and the new Nova family of foundation models.",
    "These models are designed to deliver low latency and high throughput at a fraction of the cost.",
    "We've been working with customers across many industries to bring generative AI into production.",
    "Our team builds some of the largest internet services on the planet, and many of them run on AWS.",
    # 長文（25語〜）: ~2-3s 目標
    "When we started this project, we had a simple goal in mind: to make machine learning accessible "
    "to every developer, regardless of their background or experience with AI.",
    "Homomorphic encryption allows computation on encrypted data without ever decrypting it, which has "
    "significant implications for privacy-preserving machine learning in the cloud.",
    "We're driven by the idea that the products and services we create should help people unleash their "
    "creativity and potential, and that's why we partner closely with AWS to scale globally.",
]


def _percentile(data: list[float], pct: float) -> float:
    """ソート済み前提でpct位（0〜100）を返す."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _bench_one_model(model: str, samples: list[str], runs: int) -> dict:
    """1モデルで全サンプルを計測し、統計値を返す."""
    # モデル切替: グローバル定数を上書きしてからクライアントを生成
    original_model = translator.OLLAMA_TRANSLATE_MODEL
    original_backend = translator.TRANSLATION_BACKEND
    translator.TRANSLATION_BACKEND = "ollama"
    translator.OLLAMA_TRANSLATE_MODEL = model
    try:
        client = translator.create_translate_client()

        # ウォームアップ: モデルをメモリにロードする
        print(f"  [{model}] warmup...", end="", flush=True)
        warmup_start = time.perf_counter()
        translator.translate_text(samples[0], "en", "ja", client)
        warmup_elapsed = time.perf_counter() - warmup_start
        print(f" {warmup_elapsed:.2f}s")

        # 計測本番
        print(f"  [{model}] running {len(samples)} samples x {runs} run(s)...")
        results: list[tuple[str, float, str]] = []
        for sample in samples:
            sample_times: list[float] = []
            translation = ""
            for _ in range(runs):
                t0 = time.perf_counter()
                translation = translator.translate_text(sample, "en", "ja", client)
                sample_times.append(time.perf_counter() - t0)
            # 同サンプル複数回はmedianを採用
            results.append(
                (sample, statistics.median(sample_times), translation)
            )

        elapsed = [r[1] for r in results]
        return {
            "model": model,
            "warmup": warmup_elapsed,
            "results": results,
            "min": min(elapsed),
            "max": max(elapsed),
            "mean": statistics.mean(elapsed),
            "median": statistics.median(elapsed),
            "p95": _percentile(elapsed, 95),
        }
    finally:
        translator.OLLAMA_TRANSLATE_MODEL = original_model
        translator.TRANSLATION_BACKEND = original_backend


def _print_per_sample(results: list[tuple[str, float, str]]) -> None:
    """1サンプルごとの結果をターミナルに表示."""
    print(f"  {'sec':>6}  {'words':>5}  sample")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 60}")
    for sample, elapsed, translation in results:
        word_count = len(sample.split())
        # サンプルは50文字でtruncate
        sample_disp = sample if len(sample) <= 50 else sample[:47] + "..."
        print(f"  {elapsed:>6.2f}  {word_count:>5}  {sample_disp}")
        print(f"  {'':>6}  {'':>5}  → {translation[:80]}")


def _print_summary(stats: dict) -> None:
    """統計値サマリを表示."""
    print()
    print(f"  === {stats['model']} ===")
    print(f"  warmup: {stats['warmup']:.2f}s (load + first inference)")
    print(
        f"  min: {stats['min']:.2f}s  median: {stats['median']:.2f}s  "
        f"mean: {stats['mean']:.2f}s  p95: {stats['p95']:.2f}s  max: {stats['max']:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="計測するOllamaモデル。未指定時はOLLAMA_TRANSLATE_MODELを使用",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="複数モデルを順に比較する場合に指定（例: --models gemma3:4b gemma4:e4b）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="同一サンプルの実行回数（中央値を採用）。デフォルト1",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="サンプルごとの詳細表示を抑制し、サマリだけ出す",
    )
    args = parser.parse_args()

    if args.models:
        target_models = args.models
    elif args.model:
        target_models = [args.model]
    else:
        target_models = [translator.OLLAMA_TRANSLATE_MODEL]

    print(f"Backend: ollama  |  Host: {translator.OLLAMA_HOST}")
    print(f"Models: {', '.join(target_models)}")
    print(f"Samples: {len(SAMPLES)}  Runs per sample: {args.runs}")
    print()

    all_stats: list[dict] = []
    for model in target_models:
        try:
            stats = _bench_one_model(model, SAMPLES, args.runs)
        except Exception as exc:
            print(f"  [{model}] FAILED: {exc}")
            continue
        if not args.no_detail:
            _print_per_sample(stats["results"])
        _print_summary(stats)
        all_stats.append(stats)

    # 複数モデル比較サマリ
    if len(all_stats) > 1:
        print()
        print("=== Comparison ===")
        print(
            f"  {'model':<30}  {'median':>7}  {'mean':>7}  {'p95':>7}  {'max':>7}"
        )
        print(f"  {'-' * 30}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
        for s in all_stats:
            print(
                f"  {s['model']:<30}  "
                f"{s['median']:>6.2f}s  {s['mean']:>6.2f}s  "
                f"{s['p95']:>6.2f}s  {s['max']:>6.2f}s"
            )


if __name__ == "__main__":
    main()
