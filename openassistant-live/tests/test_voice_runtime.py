"""Exercise the real decoder, bundled VAD and native channel-layout regression offline."""

import subprocess
import sys
import wave

import av
import numpy as np
import pytest
from faster_whisper.audio import decode_audio
from faster_whisper.vad import get_speech_timestamps


@pytest.mark.parametrize("suffix", [".wav", ".ogg"])
def test_real_voice_decoder_matches_pinned_whisper_dependency(tmp_path, suffix):
    wav = tmp_path / "silence.wav"
    with wave.open(str(wav), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)

    path = wav
    if suffix == ".ogg":
        path = tmp_path / "silence.ogg"
        with av.open(str(wav)) as source, av.open(str(path), "w") as target:
            stream = target.add_stream("libopus", rate=48000)
            stream.layout = "mono"
            for frame in source.decode(audio=0):
                for packet in stream.encode(frame):
                    target.mux(packet)
            for packet in stream.encode():
                target.mux(packet)

    decoded = decode_audio(str(path))
    assert decoded.shape == (1600,)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, 0)


def test_bundled_voice_activity_detector_runs_without_model_downloads():
    assert get_speech_timestamps(np.zeros(16000, dtype=np.float32)) == []


def test_custom_channel_layout_copy_does_not_double_free():
    # Isolate the native crash so an affected PyAV fails this test instead of pytest.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
            "import av; original = av.AudioLayout('FR+FL'); "
            "copied = av.AudioLayout(original); del original; del copied",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
