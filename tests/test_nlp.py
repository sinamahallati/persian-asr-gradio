from persian_asr.nlp import normalize_fa, sentiment_score


def test_sentiment_mapping_range():
    label, score = sentiment_score("این یک تست خیلی خوب است")
    assert label in {"positive", "negative", ""} 
    assert -1.0 <= score <= 1.0

def test_normalize_not_crash():
    out = normalize_fa("سلام!!!   دنیا")
    assert isinstance(out, str)
    assert out.strip() != ""