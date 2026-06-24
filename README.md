# 1. Instalar dependencias
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
pip install sounddevice soundfile resampy

# 2. Ejecutar
python nemotron_mac.py                    # micrófono, auto-detect idioma
python nemotron_mac.py --lang es-ES       # forzar español
python nemotron_mac.py --chunk 560        # más precisión, más latencia
python nemotron_mac.py --file audio.wav   # transcribir archivo
