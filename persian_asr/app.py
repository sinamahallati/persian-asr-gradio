from __future__ import annotations

import logging

import gradio as gr
from gradio.themes import Soft

from .asr import transcribe_file, tts_persian
from .nlp import normalize_fa, sentiment_score

logger = logging.getLogger("persian_asr")

theme = Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700'
'&display=swap');
:root { --app-font: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI',
Roboto, 'Helvetica Neue', Arial, 'Noto Sans'; }
.gradio-container { max-width: 900px; margin: 0 auto; font-family: var(--app-font); }

/* Hero */
.hero {
  background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 60%, #EC4899 120%);
  color: white; border-radius: 18px; padding: 20px 22px; margin: 8px 0 18px;
  box-shadow: 0 16px 40px rgba(99,102,241,.25);
}
.hero h1 { margin: 0; letter-spacing: .2px; font-weight: 700; }

/* Panels (stacked) */
.panel {
  background: rgba(255,255,255,.65); border-radius: 16px; backdrop-filter: blur(6px);
  box-shadow: 0 8px 28px rgba(0,0,0,.08); padding: 16px; margin-bottom: 14px;
}
.dark .panel { background: rgba(30,30,35,.55); }

/* Sentiment (no emoji) */
.senti { padding: 6px 8px 12px; border-radius: 14px; }
.senti-head { display: flex; align-items: center; gap: 10px; font-weight: 600; }
.senti-chip {
  font-size: 12px; padding: 2px 8px; border-radius: 999px; opacity: .95;
  background: color-mix(in srgb, #111 14%, transparent);
}
.dark .senti-chip {
  background: color-mix(in srgb, #fff 20%, transparent);
}
.senti-bar-wrap { margin-top: 10px; position: relative; height: 12px; }
.senti-bar {
  width: 100%; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #22c55e 100%);
  opacity: .9;
}
.senti-marker {
  position: absolute; top: -3px; width: 0; height: 18px; border-left: 2px solid #111;
}
.dark .senti-marker { border-left-color: #fff; }

/* Footer cleanup: hide "Use via API" & "Built with Gradio", keep "Settings" */
footer { visibility: visible }
footer a[href*="gradio_api"] { display: none !important; }   /* Use via API */
footer a[href*="gradio.app"] { display: none !important; }   /* Built with Gradio */
"""

PROGRESS_DEFAULT = gr.Progress()


def render_sentiment(label: str, score: float) -> str:
    label_clean = label.title() if label else "Neutral"
    pct = max(0.0, min(100.0, (score + 1.0) * 50.0))
    chip = f"{label_clean} ({score:+.2f})"
    return (
        "<div class='senti'>"
        "<div class='senti-head'>"
        "<span>Sentiment</span>"
        f"<span class='senti-chip'>{chip}</span>"
        "</div>"
        "<div class='senti-bar-wrap'>"
        "<div class='senti-bar'></div>"
        f"<div class='senti-marker' style='left:{pct}%;'></div>"
        "</div>"
        "</div>"
    )


class WebApp:
    def __init__(self, title: str = "Persian ASR"):
        self.title = title

    def _ui_stt(self) -> None:
        with gr.Tab("Speech → Text (ASR)"):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Input")
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )
                use_vad = gr.Checkbox(
                    value=True, label="Use VAD (auto segmentation)"
                )
                run_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Output")
                txt_raw = gr.Textbox(label="Raw Transcript", lines=6)
                txt_norm = gr.Textbox(label="Normalized", lines=6)
                senti_html = gr.HTML(label="Sentiment")

            def _run(path, use_vad_flag, progress=PROGRESS_DEFAULT):
                if not path:
                    return "", "", render_sentiment("", 0.0)
                progress = progress if callable(progress) else PROGRESS_DEFAULT
                progress(0.15, desc="Transcribing…")
                text = transcribe_file(path, use_vad=use_vad_flag)
                progress(0.6, desc="Normalizing…")
                norm = normalize_fa(text) if text else ""
                label, score = sentiment_score(norm) if norm else ("", 0.0)
                html = render_sentiment(label, score)
                progress(1.0, desc="Done")
                return text, norm, html

            run_btn.click(
                _run, inputs=[audio, use_vad], outputs=[txt_raw, txt_norm, senti_html]
            )

    def _ui_tts(self) -> None:
        with gr.Tab("Text → Speech (TTS)"):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Input")
                txt = gr.Textbox(lines=4, label="Text (Persian)")
                speak_btn = gr.Button("Synthesize", variant="primary")
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("### Output")
                out_audio = gr.Audio(label="Generated Speech", type="numpy")

            def _speak(t, progress=PROGRESS_DEFAULT):
                if not t:
                    return None
                progress = progress if callable(progress) else PROGRESS_DEFAULT
                progress(0.25, desc="Synthesizing…")
                sr, wav = tts_persian(t)
                progress(1.0, desc="Done")
                return (sr, wav)

            speak_btn.click(_speak, inputs=txt, outputs=out_audio)

    def launch(
        self,
        share: bool = False,
        inbrowser: bool = True,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        **kwargs,
    ) -> None:
        with gr.Blocks(title=self.title, css=CSS, theme=theme) as demo:
            hero = (
                "<div class='hero'>"
                f"<h1>{self.title}</h1>"
                "</div>"
            )
            js = (
                "<script>(()=>{"
                "const kill=()=>{"
                "document.querySelectorAll('footer a').forEach(a=>{"
                "const t=(a.textContent||'').toLowerCase();"
                "if(t.includes('built with gradio')||t.includes('use via api')){"
                "a.style.display='none';}"
                "});};kill();setInterval(kill,1000);})();</script>"
            )
            gr.HTML(hero + js)
            self._ui_stt()
            self._ui_tts()
        demo.launch(
            share=share,
            inbrowser=inbrowser,
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )
