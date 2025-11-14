import argparse
import logging
import sys

from .app import WebApp
from .asr import transcribe_file, tts_persian

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(prog="persian-asr")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_asr = sub.add_parser("asr", help="Speech to text (ASR)")
    p_asr.add_argument(
        "audio_path",
        help="Path to an audio file ",
    )

    p_tts = sub.add_parser("tts", help="Text to speech (TTS)")
    p_tts.add_argument("text", nargs="+", help="Persian text to synthesize")

    p_ui = sub.add_parser("launch", help="Launch the Gradio web UI")
    p_ui.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    p_ui.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the browser",
    )
    p_ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (e.g., 0.0.0.0)",
    )
    p_ui.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )

    args = parser.parse_args()

    if args.cmd == "asr":
        print(transcribe_file(args.audio_path))
    elif args.cmd == "tts":
        sr, wav = tts_persian(" ".join(args.text))
        print(f"Generated audio: sample_rate={sr}, samples={len(wav)}")
    elif args.cmd == "launch":
        WebApp().launch(
            share=args.share,
            inbrowser=not args.no_open,
            server_name=args.host,
            server_port=args.port,
        )


if __name__ == "__main__":
    sys.exit(main())
