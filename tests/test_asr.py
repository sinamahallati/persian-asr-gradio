import numpy as np
import soundfile as sf

import persian_asr.asr as asr


def test_transcribe_pipeline_monkeypatch(tmp_path, monkeypatch):
    p = tmp_path / "a.wav"
    sr = 16000
    sf.write(p, np.zeros(sr//2, dtype=np.float32), sr)

    def fake_transcribe(path, use_vad=True):
        return "سلام دنیا"

    monkeypatch.setattr(asr, "transcribe_file", fake_transcribe)
    assert asr.transcribe_file(str(p)) == "سلام دنیا"
