"""Модуль для расчёта метрик громкости аудиофайла."""

import logging

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly

from ..audio.loader import AudioData

logger = logging.getLogger(__name__)

SHORT_TERM_WINDOW = 3  # Длина окна в секундах
SHORT_TERM_STEP = 1    # Шаг между окнами в секундах


def _calculate_short_term_lufs(
    samples: np.ndarray,
    sample_rate: int,
) -> list[float | None]:
    """Рассчитывает Short-term LUFS по 3-секундным окнам."""
    window_size = SHORT_TERM_WINDOW * sample_rate
    step = SHORT_TERM_STEP * sample_rate

    if len(samples) < window_size:
        return []

    meter = pyln.Meter(sample_rate)
    results = []

    for start in range(0, len(samples) - window_size + 1, step):
        window = samples[start:start + window_size]
        loudness = meter.integrated_loudness(window)

        results.append(
            float(loudness) if np.isfinite(loudness) else None
        )

    return results


def analyze_loudness(audio: AudioData) -> dict:
    """Рассчитывает основные метрики громкости аудиофайла.

    Args:
        audio: Объект с аудиосэмплами и частотой дискретизации.

    Returns:
        Словарь с метриками:
        - integrated_lufs: интегральная громкость записи;
        - short_term_lufs: громкость на трёхсекундных окнах;
        - true_peak: оценка максимального пика с передискретизацией;
        - sample_peak: максимальная амплитуда исходных сэмплов;
        - rms: среднеквадратичная амплитуда.
    """
    logger.info("Начинаем анализ громкости: %s", audio.path.name)

    samples = np.asarray(audio.samples, dtype=np.float64)
    sample_rate = audio.sample_rate

    if sample_rate <= 0 or samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("Некорректные аудиоданные.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("Аудиоданные содержат некорректные значения.")

    # Integrated LUFS
    meter = pyln.Meter(sample_rate)
    integrated_lufs = meter.integrated_loudness(samples)

    if not np.isfinite(integrated_lufs):
        integrated_lufs = None

    # Short-term LUFS
    short_term_lufs = _calculate_short_term_lufs(
        samples, sample_rate
    )

    # Sample Peak
    sample_peak = float(np.max(np.abs(samples)))

    # True Peak (передискретизация в 4 раза)
    oversampled = resample_poly(samples, 4, 1, axis=0)
    true_peak = float(np.max(np.abs(oversampled)))

    # RMS
    rms = float(np.sqrt(np.mean(samples ** 2)))

    metrics = {
        "integrated_lufs": integrated_lufs,
        "short_term_lufs": short_term_lufs,
        "true_peak": true_peak,
        "sample_peak": sample_peak,
        "rms": rms,
    }

    logger.info("Анализ громкости завершён: %s", audio.path.name)

    return metrics
