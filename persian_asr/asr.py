from __future__ import annotations

import logging
import re
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model

from .decorators import asr_logged
from .vad import split_on_speech

logger = logging.getLogger("persian_asr")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DTYPE = torch.float32 if DEVICE.type in {"mps", "cpu"} else torch.float16

MODEL_ID = "facebook/seamless-m4t-v2-large"
TARGET_LANG = "pes"
ASR_SR = 16000

_processor = None
_model = None


def _ensure_model_loaded() -> None:
    global _processor, _model
    if _processor is None or _model is None:
        logger.info("Loading model %s on %s (dtype=%s)...", MODEL_ID, DEVICE, DTYPE)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = SeamlessM4Tv2Model.from_pretrained(
            MODEL_ID, dtype=DTYPE
        ).to(DEVICE).eval()


def _load_audio_to_16k_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != ASR_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=ASR_SR)
    wav = wav.astype(np.float32, copy=False)
    m = float(np.max(np.abs(wav))) if wav.size else 1.0
    if m > 1.0:
        wav = wav / m
    return wav


def _strip_silence(wav: np.ndarray, top_db: int = 30) -> np.ndarray:
    """حذف سکوت‌های بلند ابتدا/انتها برای جلوگیری از لوپ مدل."""
    if wav.size == 0:
        return wav
    intervals = librosa.effects.split(
        wav, top_db=top_db, frame_length=2048, hop_length=512
    )
    if len(intervals) == 0:
        return wav
    start = intervals[0][0]
    end = intervals[-1][1]
    return wav[start:end]


def _chunk_audio(seg: np.ndarray, max_seconds: int = 10) -> List[np.ndarray]:
    max_len = ASR_SR * max_seconds
    if len(seg) <= max_len:
        return [seg]
    return [seg[i:i + max_len] for i in range(0, len(seg), max_len)]


_DIAC = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8"
    r"\u06EA-\u06ED\u200C\u200F\u200E\u0640]"
)


def _clean_tail(text: str, keep: int = 1) -> str:
    base = _DIAC.sub("", text)
    toks = re.findall(r"\S+", base)
    if len(toks) < 8:
        return text

    def _cut(n: int, toks: list[str]) -> int:
        if len(toks) < n * 2:
            return len(toks)
        pat = toks[-n:]
        i = len(toks) - 2 * n
        reps = 0
        while i >= 0 and toks[i:i + n] == pat:
            reps += 1
            i -= n
        if reps > keep:
            return len(toks) - n * (reps - keep)
        return len(toks)

    cut = min(_cut(2, toks), _cut(1, toks))
    if cut == len(toks):
        return text
    idx = 0
    out = []
    for m in re.finditer(r"\S+", text):
        if idx >= cut:
            break
        out.append(m.group())
        idx += 1
    return " ".join(out)


def _decode_generated(generated) -> str:
    """Tensor/Dict → list[int] → decode."""
    seq = generated.sequences if hasattr(generated, "sequences") else generated
    if isinstance(seq, torch.Tensor):
        ids = seq[0].detach().cpu().tolist()
    else:
        first = seq[0]
        ids = first.tolist() if hasattr(first, "tolist") else list(first)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return _processor.tokenizer.decode(ids, skip_special_tokens=True).strip()


@asr_logged
def transcribe_file(
    path: str,
    use_vad: bool = True,
    vad_aggressiveness: int = 3, 
    chunk_seconds: int = 10, 
) -> str:

    _ensure_model_loaded()
    wav = _strip_silence(_load_audio_to_16k_mono(path))
    segments = (
        split_on_speech(wav, sr=ASR_SR, aggressiveness=vad_aggressiveness)
        if use_vad else [wav]
    )

    tok = _processor.tokenizer
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", eos_id)

    texts: List[str] = []
    for seg in segments:
        for chunk in _chunk_audio(_strip_silence(seg), max_seconds=chunk_seconds):
            inputs = _processor(
                audio=chunk, sampling_rate=ASR_SR, return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = _model.generate(
                    **inputs,
                    tgt_lang=TARGET_LANG,
                    generate_speech=False,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=160,
                    no_repeat_ngram_size=6,   
                    repetition_penalty=1.12, 
                    early_stopping=True,
                    length_penalty=0.8,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    return_dict_in_generate=True,
                )
            text = _decode_generated(out)
            if text:
                texts.append(text)

    full = " ".join(texts).strip()
    return _clean_tail(full, keep=1)


@asr_logged
def tts_persian(text: str) -> Tuple[int, np.ndarray]:
    _ensure_model_loaded()
    if not text:
        return ASR_SR, np.zeros(0, dtype=np.float32)

    inputs = _processor(text=text, src_lang=TARGET_LANG, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        audio = _model.generate(
            **inputs, tgt_lang=TARGET_LANG, do_sample=False, num_beams=1
        )[0].detach().cpu().numpy().squeeze()

    sr = getattr(_model.config, "sampling_rate", 16000)
    return sr, audio.astype(np.float32, copy=False)
