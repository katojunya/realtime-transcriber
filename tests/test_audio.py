"""AudioCaptureクラスとfind_device関数のテスト."""

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from realtime_transcriber.audio import (
    AudioCapture,
    find_device,
)


class TestFindDevice:
    """find_device関数のテスト."""

    def test_should_return_device_index_when_exact_name_matches(self) -> None:
        # Given: BlackHole 2chが登録されたデバイスリスト
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]

        # When: BlackHole 2chを検索する
        index = find_device("BlackHole 2ch", mock_sd)

        # Then: 正しいインデックスが返る
        assert index == 1

    def test_should_return_device_index_when_partial_name_matches(self) -> None:
        # Given: 名前に部分一致するデバイス
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "BlackHole 2ch (Virtual)", "max_input_channels": 2},
        ]

        # When: 部分一致で検索する
        index = find_device("BlackHole 2ch", mock_sd)

        # Then: 部分一致したデバイスのインデックスが返る
        assert index == 1

    def test_should_raise_error_when_device_not_found(self) -> None:
        # Given: 対象デバイスが存在しないリスト
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
        ]

        # When/Then: 存在しないデバイスを検索するとエラー
        with pytest.raises(RuntimeError):
            find_device("BlackHole 2ch", mock_sd)

    def test_should_raise_error_when_no_devices_available(self) -> None:
        # Given: デバイスリストが空
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = []

        # When/Then: デバイス検索でエラー
        with pytest.raises(RuntimeError):
            find_device("BlackHole 2ch", mock_sd)

    def test_should_only_match_input_capable_devices(self) -> None:
        # Given: 入力チャンネルが0のデバイス（出力専用）
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 0},
            {"name": "BlackHole 2ch Input", "max_input_channels": 2},
        ]

        # When: 入力デバイスとして検索する
        index = find_device("BlackHole 2ch", mock_sd)

        # Then: 入力チャンネルを持つデバイスが返る
        assert index == 1


def _make_capture(mock_sd: MagicMock | None = None) -> AudioCapture:
    """テスト用のAudioCaptureインスタンスを生成するヘルパー."""
    if mock_sd is None:
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
    return AudioCapture(
        device_name="BlackHole 2ch",
        sample_rate=16000,
        sd_module=mock_sd,
    )


class TestAudioCaptureInit:
    """AudioCaptureのコンストラクタのテスト."""

    def test_should_store_sample_rate(self) -> None:
        # Given/When: AudioCaptureを生成する
        capture = _make_capture()

        # Then: サンプルレートが保持される
        assert capture.sample_rate == 16000


class TestAudioCaptureContextManager:
    """AudioCaptureのコンテキストマネージャのテスト."""

    def test_should_open_stream_on_enter(self) -> None:
        # Given: モック化されたsounddevice
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        capture = _make_capture(mock_sd)

        # When: コンテキストマネージャに入る
        capture.__enter__()

        # Then: InputStreamが作成・開始される
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_should_close_stream_on_exit(self) -> None:
        # Given: ストリームが開始済みのAudioCapture
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        capture = _make_capture(mock_sd)
        capture.__enter__()

        # When: コンテキストマネージャを抜ける
        capture.__exit__(None, None, None)

        # Then: ストリームが停止・クローズされる
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


class TestAudioCaptureGetAudioChunk:
    """get_audio_chunk メソッドのテスト."""

    def test_should_return_none_when_queue_is_empty(self) -> None:
        # Given: キューが空のAudioCapture
        capture = _make_capture()

        # When: 音声チャンクを取得する
        chunk = capture.get_audio_chunk()

        # Then: データがないのでNoneが返る
        assert chunk is None

    def test_should_return_none_during_speech(self) -> None:
        # Given: VADが発話中と判定する音声データ
        capture = _make_capture()
        # VADが高い確率（発話）を返すようにモック
        capture._vad = MagicMock()
        capture._vad.window_size_samples = 512
        capture._vad.process.return_value = 0.9

        stereo = np.random.rand(512, 2).astype(np.float32)
        capture._queue.put(stereo)

        # When: 発話中に取得を試みる
        chunk = capture.get_audio_chunk()

        # Then: 発話が終了していないのでNoneが返る（蓄積中）
        assert chunk is None
        assert capture._in_speech is True

    def test_should_return_audio_after_speech_then_silence(self) -> None:
        # Given: 発話→無音の遷移をシミュレート
        capture = _make_capture()
        window_size = 512
        capture._vad = MagicMock()
        capture._vad.window_size_samples = window_size
        capture._window_size = window_size
        # 最低発話長を満たすために十分な発話ウィンドウ数を計算
        min_speech_windows = (capture._min_speech_samples // window_size) + 1
        # 無音が十分続く数
        min_silence_windows = (capture._min_silence_samples // window_size) + 1

        # 発話 → 無音 の確率を返す
        probs = [0.9] * min_speech_windows + [0.0] * min_silence_windows
        capture._vad.process.side_effect = probs

        # 全ウィンドウ分のステレオデータをキューに投入
        total_samples = window_size * len(probs)
        stereo = np.random.rand(total_samples, 2).astype(np.float32)
        capture._queue.put(stereo)

        # When: 発話終了後に取得
        with patch("builtins.print"):
            chunk = capture.get_audio_chunk()

        # Then: モノラルfloat32の音声データが返る
        assert chunk is not None
        assert chunk.ndim == 1
        assert chunk.dtype == np.float32


class TestToMono:
    """_to_monoメソッドのテスト."""

    def test_should_convert_stereo_to_mono_via_mean(self) -> None:
        # Given: ステレオ音声（左=1.0, 右=0.0）
        capture = _make_capture()
        stereo = np.zeros((100, 2), dtype=np.float32)
        stereo[:, 0] = 1.0

        # When: モノラル変換する
        mono = capture._to_mono(stereo)

        # Then: 両チャンネルの平均値になる
        assert mono.ndim == 1
        assert mono.dtype == np.float32
        np.testing.assert_allclose(mono, 0.5)

    def test_should_pass_through_mono_input(self) -> None:
        # Given: モノラル音声
        capture = _make_capture()
        mono_input = np.ones(100, dtype=np.float32) * 0.5

        # When: モノラル変換する（既にモノラル）
        result = capture._to_mono(mono_input)

        # Then: そのまま返る
        assert result.ndim == 1
        np.testing.assert_allclose(result, 0.5)


class TestAudioCaptureCallback:
    """InputStreamコールバックの振る舞いテスト."""

    def test_should_copy_indata_to_queue(self) -> None:
        # Given: AudioCaptureのコールバック
        capture = _make_capture()
        capture._queue = queue.Queue()
        indata = np.ones((1024, 2), dtype=np.float32)

        # When: コールバックが呼ばれる
        capture._audio_callback(indata, frames=1024, time_info=None, status=None)

        # Then: キューにコピーされたデータが入る
        queued = capture._queue.get_nowait()
        np.testing.assert_array_equal(queued, indata)
        # コピーであることを確認（元データと同一オブジェクトでない）
        assert queued is not indata
