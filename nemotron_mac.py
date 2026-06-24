#!/usr/bin/env python3
"""
Nemotron 3.5 ASR — Mac, streaming en tiempo real
Cache-Aware FastConformer-RNNT con detección automática de idioma

Uso:
    python nemotron_mac.py                    # micrófono, auto-detect
    python nemotron_mac.py --lang es-ES       # forzar idioma
    python nemotron_mac.py --chunk 320        # 80/160/320/560/1120 ms
    python nemotron_mac.py --file audio.wav   # archivo
"""

import argparse
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── Silenciar logs verbosos de NeMo ───────────────────────────────────────────
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("nemo").setLevel(logging.ERROR)

# Silenciar el logger específico de NeMo que imprime las configs de train/val/test
for _name in logging.root.manager.loggerDict:
    if "nemo" in _name.lower():
        logging.getLogger(_name).setLevel(logging.ERROR)

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH = str(
    Path.home()
    / ".cache/huggingface/hub"
    / "models--nvidia--nemotron-3.5-asr-streaming-0.6b"
    / "snapshots/3fc30f3e2ae5d78d462441f3ce89dda694f89bd7"
    / "nemotron-3.5-asr-streaming-0.6b.nemo"
)

SAMPLE_RATE = 16_000

CHUNK_TO_CONTEXT = {
    80:   [56, 0],
    160:  [56, 1],
    320:  [56, 3],
    560:  [56, 6],
    1120: [56, 13],
}

LANG_EMOJI = {
    "en": "🇺🇸", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹",
    "pt": "🇧🇷", "nl": "🇳🇱", "ru": "🇷🇺", "ar": "🇸🇦", "hi": "🇮🇳",
    "ja": "🇯🇵", "ko": "🇰🇷", "vi": "🇻🇳", "uk": "🇺🇦", "tr": "🇹🇷",
    "zh": "🇨🇳", "pl": "🇵🇱", "sv": "🇸🇪", "cs": "🇨🇿", "da": "🇩🇰",
    "fi": "🇫🇮", "nb": "🇳🇴", "bg": "🇧🇬", "hr": "🇭🇷", "sk": "🇸🇰",
    "ro": "🇷🇴", "et": "🇪🇪", "hu": "🇭🇺",
}

LANG_TAG_RE = re.compile(r"<([a-z]{2}-[A-Z]{2})>\s*$")


def flag(lang_tag: str) -> str:
    code = lang_tag.split("-")[0].lower()
    return LANG_EMOJI.get(code, "🌐")


# ── Extraer texto de Hypothesis o string ───────────────────────────────────────

def extract_text(obj) -> str:
    """
    model.transcribe() puede devolver:
      - str
      - Hypothesis (tiene .text)
      - lista de cualquiera de los anteriores
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return extract_text(obj[0]) if obj else ""
    # Hypothesis / NBestHypotheses / cualquier objeto con .text
    if hasattr(obj, "text"):
        return str(obj.text)
    if hasattr(obj, "y_sequence"):          # último recurso
        return str(obj.y_sequence)
    return str(obj)


def parse_output(raw_obj) -> tuple[str, str | None]:
    """Devuelve (texto_limpio, lang_tag | None)."""
    raw = extract_text(raw_obj).strip()
    m = LANG_TAG_RE.search(raw)
    if m:
        return raw[: m.start()].strip(), m.group(1)
    return raw, None


# ── Carga del modelo ───────────────────────────────────────────────────────────

def load_model(chunk_ms: int):
    print("⏳ Cargando NeMo…")
    import os
    os.environ.setdefault("NEMO_TESTING", "1")  # suprime algunos prints extra

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit("❌ pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'")

    # NeMo registra sus loggers al importar → silenciarlos después del import
    for _ln in list(logging.root.manager.loggerDict):
        if any(x in _ln.lower() for x in ("nemo", "lightning", "pytorch_lightning")):
            logging.getLogger(_ln).setLevel(logging.ERROR)

    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cpu")
    model.eval()

    # Configurar chunk size
    ctx = CHUNK_TO_CONTEXT.get(chunk_ms, [56, 3])
    try:
        model.change_attention_model(att_context_size=ctx)
    except Exception:
        try:
            model.encoder.set_default_att_context_size(ctx)
        except Exception:
            pass  # algunos builds configuran esto internamente

    print(f"✅ Modelo listo  chunk={chunk_ms}ms  context={ctx}\n")
    return model


# ── Wrapper de transcripción ───────────────────────────────────────────────────

def transcribe_audio(model, audio: np.ndarray, target_lang: str) -> tuple[str, str | None]:
    """audio: float32 mono @ 16kHz"""
    try:
        # Intentar con target_lang kwarg (NeMo reciente)
        result = model.transcribe(
            [audio],
            batch_size=1,
            target_lang=target_lang,
            return_hypotheses=False,    # forzar strings
        )
    except TypeError:
        try:
            result = model.transcribe(
                [audio],
                batch_size=1,
                return_hypotheses=False,
            )
        except TypeError:
            result = model.transcribe([audio], batch_size=1)

    return parse_output(result[0] if isinstance(result, list) else result)


# ── Streaming en tiempo real con VAD simple ────────────────────────────────────

class RealtimeTranscriber:
    """
    Acumula audio del micrófono en un buffer.
    Cada vez que el buffer llega a chunk_ms, lo manda al modelo en un thread
    separado para no bloquear la captura de audio.
    """

    def __init__(self, model, chunk_ms: int, target_lang: str):
        self.model = model
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.target_lang = target_lang
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.running = True
        self.silence_rms = 0.008
        # Buffer acumulador para detección de silencio más robusta
        self._buffer = np.array([], dtype=np.float32)

    def audio_callback(self, indata, frames, time_info, status):
        """Llamado por sounddevice en cada bloque capturado (~10ms)."""
        chunk = indata[:, 0].copy()
        self._buffer = np.concatenate([self._buffer, chunk])

        # Cuando acumulamos suficiente audio, mandarlo a la cola
        while len(self._buffer) >= self.chunk_samples:
            segment = self._buffer[: self.chunk_samples]
            self._buffer = self._buffer[self.chunk_samples :]
            self.audio_queue.put(segment)

    def inference_worker(self):
        """Thread dedicado a inferencia para no bloquear el audio."""
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Saltar silencio
            rms = np.sqrt(np.mean(audio**2))
            if rms < self.silence_rms:
                continue

            t0 = time.perf_counter()
            text, lang = transcribe_audio(self.model, audio, self.target_lang)
            latency_ms = (time.perf_counter() - t0) * 1000

            if text.strip():
                lang_display = lang or self.target_lang
                emoji = flag(lang_display)
                # \r + ANSI para sobreescribir línea si es el mismo idioma seguido
                print(f"{emoji} [{lang_display}]  {text}  \033[90m({latency_ms:.0f}ms)\033[0m")

    def run(self):
        try:
            import sounddevice as sd
        except ImportError:
            sys.exit("❌ pip install sounddevice")

        # Lanzar thread de inferencia
        worker = threading.Thread(target=self.inference_worker, daemon=True)
        worker.start()

        print("🎙️  Escuchando en tiempo real… (Ctrl+C para salir)")
        print("─" * 60)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(SAMPLE_RATE * 0.01),   # 10ms bloques de captura
                callback=self.audio_callback,
            ):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
            print("\n\n👋 Detenido.")


# ── Modo archivo ───────────────────────────────────────────────────────────────

def run_file(model, filepath: str, chunk_ms: int, target_lang: str):
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("❌ pip install soundfile")

    print(f"📂 Transcribiendo: {filepath}")
    print("─" * 60)

    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    if sr != SAMPLE_RATE:
        try:
            import resampy
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        except ImportError:
            print(f"⚠️  Audio a {sr}Hz, necesita 16kHz.")
            print(f"   ffmpeg -i {filepath} -ar 16000 -ac 1 output_16k.wav")
            sys.exit(1)

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    transcript_parts = []
    lang_counts: dict[str, int] = {}

    for i in range(n_chunks):
        segment = audio[i * chunk_samples : (i + 1) * chunk_samples]
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))

        text, lang = transcribe_audio(model, segment, target_lang)
        if text.strip():
            lang_display = lang or target_lang
            ts = i * chunk_ms / 1000
            print(f"[{ts:6.2f}s] {flag(lang_display)} [{lang_display}]  {text}")
            transcript_parts.append(text)
            lang_counts[lang_display] = lang_counts.get(lang_display, 0) + 1

    print("\n" + "─" * 60)
    print("📝 Transcripción completa:\n")
    print(" ".join(transcript_parts))

    if lang_counts:
        dominant = max(lang_counts, key=lang_counts.get)
        print(f"\n🌍 Idioma dominante: {flag(dominant)} {dominant}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nemotron 3.5 ASR — Mac realtime")
    parser.add_argument("--lang", default="auto",
        help="es-ES, en-US, fr-FR… o 'auto' para detección automática")
    parser.add_argument("--chunk", type=int, default=320,
        choices=[80, 160, 320, 560, 1120],
        help="Chunk en ms (menor=más latencia, más WER)")
    parser.add_argument("--file", type=str, default=None,
        help="Archivo .wav a transcribir (sin esto usa micrófono)")
    parser.add_argument("--silence", type=float, default=0.008,
        help="Umbral RMS de silencio (default 0.008)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   Nemotron 3.5 ASR — FastConformer-RNNT         ║")
    print("║   40 idiomas · Cache-Aware Streaming             ║")
    print("╚══════════════════════════════════════════════════╝\n")

    model = load_model(args.chunk)

    if args.file:
        run_file(model, args.file, args.chunk, args.lang)
    else:
        transcriber = RealtimeTranscriber(model, args.chunk, args.lang)
        transcriber.silence_rms = args.silence
        transcriber.run()


if __name__ == "__main__":
    main()
