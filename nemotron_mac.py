#!/usr/bin/env python3
"""
Nemotron 3.5 ASR Streaming — Mac
Cache-Aware FastConformer-RNNT con detección automática de idioma (40 locales)

Uso:
    python nemotron_mac.py                    # auto-detect idioma
    python nemotron_mac.py --lang es-ES       # forzar español
    python nemotron_mac.py --chunk 320        # chunk size en ms (80/160/320/560/1120)
    python nemotron_mac.py --file audio.wav   # transcribir archivo
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

# ── Configuración ──────────────────────────────────────────────────────────────

MODEL_PATH = str(
    Path.home()
    / ".cache/huggingface/hub"
    / "models--nvidia--nemotron-3.5-asr-streaming-0.6b"
    / "snapshots/3fc30f3e2ae5d78d462441f3ce89dda694f89bd7"
    / "nemotron-3.5-asr-streaming-0.6b.nemo"
)

SAMPLE_RATE = 16_000  # Hz — requerido por el modelo

# chunk_ms → att_context_size (right context en frames de 80ms)
CHUNK_TO_CONTEXT = {
    80:   [56, 0],
    160:  [56, 1],
    320:  [56, 3],
    560:  [56, 6],
    1120: [56, 13],
}

# Emojis de bandera por código de idioma (muestra en terminal)
LANG_EMOJI = {
    "en": "🇺🇸", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹",
    "pt": "🇧🇷", "nl": "🇳🇱", "ru": "🇷🇺", "ar": "🇸🇦", "hi": "🇮🇳",
    "ja": "🇯🇵", "ko": "🇰🇷", "vi": "🇻🇳", "uk": "🇺🇦", "tr": "🇹🇷",
    "zh": "🇨🇳", "pl": "🇵🇱", "sv": "🇸🇪", "cs": "🇨🇿", "da": "🇩🇰",
    "fi": "🇫🇮", "nb": "🇳🇴", "bg": "🇧🇬", "hr": "🇭🇷", "sk": "🇸🇰",
    "ro": "🇷🇴", "et": "🇪🇪", "hu": "🇭🇺",
}


def flag(lang_tag: str) -> str:
    """Devuelve emoji de bandera para un lang tag como 'en-US' o 'es-ES'."""
    code = lang_tag.split("-")[0].lower()
    return LANG_EMOJI.get(code, "🌐")


# ── Parseo de output del modelo ────────────────────────────────────────────────

LANG_TAG_RE = re.compile(r"<([a-z]{2}-[A-Z]{2})>\s*$")


def parse_output(raw: str) -> tuple[str, str | None]:
    """
    El modelo añade un tag de idioma al final del texto cuando strip_lang_tags=False.
    Ejemplo: "Hola, ¿cómo estás? <es-ES>"
    Devuelve (texto_limpio, lang_tag | None)
    """
    m = LANG_TAG_RE.search(raw.strip())
    if m:
        lang = m.group(1)
        text = raw[: m.start()].strip()
        return text, lang
    return raw.strip(), None


# ── Carga del modelo ───────────────────────────────────────────────────────────

def load_model(chunk_ms: int, target_lang: str):
    print("⏳ Cargando NeMo (puede tardar 30–60s en Mac)…")
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit(
            "❌ NeMo no está instalado.\n"
            "   Instálalo con:\n"
            "   pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
        )

    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
    model.eval()

    # Configurar att_context_size para el chunk elegido
    ctx = CHUNK_TO_CONTEXT.get(chunk_ms, [56, 3])
    if hasattr(model, "change_attention_model"):
        model.change_attention_model(att_context_size=ctx)
    elif hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(ctx)

    print(f"✅ Modelo listo — chunk={chunk_ms}ms  context={ctx}  lang={target_lang}\n")
    return model


# ── Transcripción de un chunk de audio ────────────────────────────────────────

def transcribe(model, audio: np.ndarray, target_lang: str) -> tuple[str, str | None]:
    """
    audio: float32 array @ 16kHz, mono
    Devuelve (texto, lang_tag)
    """
    # El modelo acepta target_lang como kwarg cuando está disponible
    try:
        output = model.transcribe(
            [audio],
            batch_size=1,
            target_lang=target_lang,
            strip_lang_tags=False,   # conservamos el tag para leerlo
        )
    except TypeError:
        # Versión de NeMo sin soporte de target_lang como kwarg → fallback
        output = model.transcribe([audio], batch_size=1)

    raw = output[0] if output else ""
    return parse_output(raw)


# ── Modo micrófono (streaming en tiempo real) ──────────────────────────────────

def run_microphone(model, chunk_ms: int, target_lang: str):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("❌ Instala sounddevice:  pip install sounddevice")

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    silence_threshold = 0.005  # RMS mínimo para procesar

    print(f"🎙️  Escuchando en chunks de {chunk_ms}ms… (Ctrl+C para salir)")
    print("─" * 60)

    try:
        while True:
            audio = sd.rec(
                chunk_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio_flat = audio[:, 0]

            # Saltar silencio
            if np.sqrt(np.mean(audio_flat**2)) < silence_threshold:
                continue

            t0 = time.perf_counter()
            text, lang = transcribe(model, audio_flat, target_lang)
            latency = (time.perf_counter() - t0) * 1000

            if text:
                lang_display = lang or target_lang
                emoji = flag(lang_display)
                print(f"{emoji} [{lang_display}]  {text}  \033[90m({latency:.0f}ms)\033[0m")

    except KeyboardInterrupt:
        print("\n\n👋 Detenido.")


# ── Modo archivo ───────────────────────────────────────────────────────────────

def run_file(model, filepath: str, chunk_ms: int, target_lang: str):
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("❌ Instala soundfile:  pip install soundfile")

    print(f"📂 Transcribiendo: {filepath}")
    print("─" * 60)

    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # mono

    # Resamplear si es necesario
    if sr != SAMPLE_RATE:
        try:
            import resampy
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        except ImportError:
            print(f"⚠️  El audio es {sr}Hz pero el modelo necesita 16kHz.")
            print("   Instala resampy (pip install resampy) o convierte el archivo con ffmpeg:")
            print(f"   ffmpeg -i {filepath} -ar 16000 -ac 1 output_16k.wav")
            sys.exit(1)

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    full_transcript = []
    detected_langs = {}

    for i in range(total_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        # Padding al último chunk si es más corto
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        text, lang = transcribe(model, chunk, target_lang)

        if text:
            lang_display = lang or target_lang
            emoji = flag(lang_display)
            ts = i * chunk_ms / 1000
            print(f"[{ts:6.2f}s] {emoji} [{lang_display}]  {text}")
            full_transcript.append(text)
            detected_langs[lang_display] = detected_langs.get(lang_display, 0) + 1

    print("\n" + "─" * 60)
    print("📝 Transcripción completa:\n")
    print(" ".join(full_transcript))

    if detected_langs:
        dominant = max(detected_langs, key=detected_langs.get)
        print(f"\n🌍 Idioma dominante detectado: {flag(dominant)} {dominant}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nemotron 3.5 ASR — Streaming multilingual en Mac"
    )
    parser.add_argument(
        "--lang",
        default="auto",
        help="Código de idioma (e.g. es-ES, en-US, fr-FR) o 'auto' para detección automática",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=320,
        choices=[80, 160, 320, 560, 1120],
        help="Tamaño de chunk en ms (menor = más latencia baja, más WER)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Ruta a archivo de audio .wav para transcribir (sin esto usa micrófono)",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   Nemotron 3.5 ASR — FastConformer-RNNT         ║")
    print("║   40 idiomas · Cache-Aware Streaming             ║")
    print("╚══════════════════════════════════════════════════╝\n")

    model = load_model(args.chunk, args.lang)

    if args.file:
        run_file(model, args.file, args.chunk, args.lang)
    else:
        run_microphone(model, args.chunk, args.lang)


if __name__ == "__main__":
    main()
