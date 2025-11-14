
# Persian ASR & TTS

Persian ASR & TTS is an open-source project that provides **Automatic Speech Recognition (ASR)** and **Text-to-Speech (TTS)** capabilities for the Persian language. The project leverages state-of-the-art models and a custom Persian TTS system. It includes a **Gradio-based web interface** and command-line tools for easy interaction.

## Features

- **ASR (Speech-to-Text)**: Transcribe Persian audio to text with **Voice Activity Detection (VAD)**.
- **TTS (Text-to-Speech)**: Synthesize natural-sounding Persian speech from text.
- **Web UI**: A Gradio-based interface for easy interaction with the model.
- **Text Normalization & Sentiment Analysis**: Using Shekar for Persian text preprocessing and sentiment scoring.

---

## Models Used

- **Automatic Speech Recognition (ASR)**: 
  - Model: **Seamless-M4T-v2** by Facebook, available on [Hugging Face](https://huggingface.co/facebook/seamless-m4t-v2-large).
  - This model is used for speech-to-text conversion for Persian language audio.

- **Text-to-Speech (TTS)**:
  - The TTS model is a custom solution built to synthesize Persian speech. It leverages **transformers** and the **Seamless-M4T-v2** model for generating speech from Persian text.

---

## Requirements

- Python **3.10+**
- Internet on first run (~9 GB model download)
- macOS (Apple Silicon), Linux, or Windows
- (macOS) If `soundfile` complains: `brew install libsndfile`

---

## Quick Start

### Set up the environment:

```bash
cd persian-asr

# 1) Virtual environment
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# (optional) Dev tools
pip install ruff pytest
```

### Run the App (UI):

```bash
python -m persian_asr launch
```

### Command-line Only (no UI):

```bash
# Speech to Text
python -m persian_asr asr path/to/audio.wav

# Text to Speech
python -m persian_asr tts "سلام! این یک تست است."
```

---

## Project Structure

```
persian_asr/
  __main__.py        # CLI: asr / tts / launch 
  app.py             # Gradio UI (stacked, English, custom theme)
  asr.py             # ASR core (VAD → silence trim → chunking → generate)
  vad.py             # WebRTC-VAD (with energy-based fallback)
  nlp.py             # Shekar normalizer, sentiment
  decorators.py      # logging/timing decorator
tests/
  test_asr.py
  test_nlp.py
pyproject.toml       # dependencies, ruff, pytest config
README.md            # Project documentation
LICENSE              # License (if applicable)
```

---

## Tests & Lint

### Run the linter:

```bash
ruff check .
ruff check --fix .
```

### Run tests:

```bash
python -m pytest
```

---

## Dependencies

The project uses the following dependencies:

- `torch`, `torchaudio`: For ASR and audio processing.
- `transformers`: For leveraging pre-trained models.
- `gradio`: For creating the UI.
- `shekar`: For text normalization and sentiment analysis.
- `librosa`, `soundfile`: For audio handling.
- `webrtcvad-wheels`: For voice activity detection.
