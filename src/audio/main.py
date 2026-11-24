"""Точка входа в анализатор.

В этом скрипте будет происходить сбор анализатора.
"""

# src/run_load.py
import logging
from pathlib import Path

from src.audio.loader import AudioLoader, AudioLoadError

logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parent
audio = AudioLoader().load(ROOT_DIR.parent / "example.mp3")

print(audio.samples.shape, audio.sample_rate, audio.path)

