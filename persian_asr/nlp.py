from __future__ import annotations

from typing import Tuple

from shekar import Normalizer, SentimentClassifier

_normalizer = Normalizer()
_sent = SentimentClassifier()

def normalize_fa(txt: str) -> str:
    return _normalizer(txt or "")

def sentiment_score(txt: str) -> Tuple[str, float]:
    if not txt:
        return "", 0.0
    label, conf = _sent(txt)
    score = float(conf if label == "positive" else -conf)
    return label, score
