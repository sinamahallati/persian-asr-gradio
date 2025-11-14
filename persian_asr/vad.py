from __future__ import annotations

import numpy as np

try:
    import webrtcvad
    _HAVE_WEBRTCVAD = True
except Exception:  
    webrtcvad = None
    _HAVE_WEBRTCVAD = False

def _float_to_int16(wav: np.ndarray) -> bytes:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16).tobytes()

def _frames(pcm: bytes, frame_bytes: int):
    for i in range(0, len(pcm), frame_bytes):
        chunk = pcm[i:i + frame_bytes]
        if len(chunk) < frame_bytes:
            break
        yield chunk

def split_on_speech(
    wav: np.ndarray,
    sr: int = 16000,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_speech_ms: int = 300
) -> list[np.ndarray]:
    
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pcm = _float_to_int16(wav)
    frame_bytes = int(sr * frame_ms / 1000) * 2 

    voiced_chunks = []
    cur = []

    if _HAVE_WEBRTCVAD:
        vad = webrtcvad.Vad(aggressiveness)
        for fr in _frames(pcm, frame_bytes):
            is_voiced = vad.is_speech(fr, sr)
            cur.append(fr)
            if not is_voiced and cur:
                seg = b"".join(cur)
                if len(seg) >= int(min_speech_ms / frame_ms) * frame_bytes:
                    audio = np.frombuffer(seg, dtype=np.int16).astype(np.float32) / 32768.0
                    voiced_chunks.append(audio)
                cur = []
        if cur:
            audio = np.frombuffer(b"".join(cur), dtype=np.int16).astype(np.float32) / 32768.0
            voiced_chunks.append(audio)
        return voiced_chunks

    hop = int(sr * frame_ms / 1000)
    energy = np.convolve(np.abs(wav), np.ones(hop) / hop, mode="same")
    thr = 0.04 * energy.max() if energy.size else 0.0
    mask = energy > thr
    
    idx = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(idx == 1)[0]
    ends = np.where(idx == -1)[0]

    for s, e in zip(starts, ends, strict=False):
        if (e - s) * 1000 / sr >= min_speech_ms:
            voiced_chunks.append(wav[s:e].astype(np.float32))
    if not voiced_chunks:
        voiced_chunks = [wav.astype(np.float32)]
    return voiced_chunks
