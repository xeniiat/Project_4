"""Загрузка аудиофайлов в numpy-массив.

Скрипт отвечает за загрузку файла, а также проверку требований платформы к файлу
(существование, формат, корректность данных, битрейт), по исполнении возвращает numpy-массив сэмплов.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".ogg"})

MIN_BITRATE_KBPS = 40
RECOMMENDED_BITRATE_KBPS = 192


class AudioLoadError(Exception):
    """Базовое исключение при загрузке аудиофайлов."""


class AudioFileNotFoundError(AudioLoadError):
    """Файл не найден."""


class UnsupportedFormatError(AudioLoadError):
    """Формат файла не поддерживается."""


class CorruptedAudioError(AudioLoadError):
    """Файл повреждён или не содержит корректных аудиоданных."""


class LowBitrateError(AudioLoadError):
    """Битрейт файла ниже минимально допустимого."""


class AudioData:
    """Загруженное аудио.

    Атрибуты:
        samples: массив float32 формы (frames, channels), значения в [-1, 1].
        sample_rate: частота дискретизации, Гц.
        path: путь к исходному файлу.
    """

    __slots__ = ("samples", "sample_rate", "path")


    def __init__(self, samples: np.ndarray, sample_rate: int, path: Path) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.path = path


    def __repr__(self) -> str:
        return (
            f"AudioData(samples=<ndarray shape={self.samples.shape}>, "
            f"sample_rate={self.sample_rate!r}, path={self.path!r})"
        )


class AudioLoader:
    """Проверяет аудиофайл на соответствие требованиям и загружает его в память."""

    def __init__(self, supported_extensions: set[str] | frozenset[str] | None = None) -> None:
        source = SUPPORTED_EXTENSIONS if supported_extensions is None else supported_extensions
        self._supported = frozenset(ext.lower() for ext in source)


    def load(self, path: str | Path) -> AudioData:
        """Проверяет и загружает файл.

        Raises:
            AudioFileNotFoundError: файл не найден.
            UnsupportedFormatError: формат не поддерживается.
            CorruptedAudioError: файл повреждён или не содержит аудио.
            LowBitrateError: битрейт ниже минимально допустимого.
        """
        path = Path(path)
        self._validate(path)

        try:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except sf.SoundFileError as exc:
            logger.error("Не удалось прочитать %s: %s", path.name, exc)
            raise CorruptedAudioError(
                f"Файл повреждён или не является аудио: {path.name}"
            ) from exc

        if sample_rate <= 0:
            logger.error("Некорректная частота дискретизации (%s): %s", sample_rate, path.name)
            raise CorruptedAudioError(
                f"Некорректная частота дискретизации ({sample_rate}): {path.name}"
            )
        if samples.shape[0] == 0:
            logger.error("Файл не содержит аудиоданных: %s", path.name)
            raise CorruptedAudioError(f"Файл не содержит аудиоданных: {path.name}")

        self._check_bitrate(path, frames=samples.shape[0], sample_rate=sample_rate)

        logger.info("Файл загружен: %s", path.name)
        return AudioData(samples, sample_rate, path)


    def _validate(self, path: Path) -> None:
        if not path.is_file():
            logger.error("Файл не найден: %s", path)
            raise AudioFileNotFoundError(f"Файл не найден: {path}")
        if path.suffix.lower() not in self._supported:
            logger.error("Неподдерживаемый формат %r: %s", path.suffix, path.name)
            raise UnsupportedFormatError(
                f"Формат {path.suffix!r} не поддерживается. "
                f"Доступные: {', '.join(sorted(self._supported))}"
            )


    @staticmethod
    def _check_bitrate(path: Path, frames: int, sample_rate: int) -> None:
        """Проверяет средний битрейт файла.

        Битрейт оценивается как размер файла / длительность, так что это
        приблизительный результат.
        """
        duration_sec = frames / sample_rate
        bitrate_kbps = path.stat().st_size * 8 / duration_sec / 1000

        if bitrate_kbps < MIN_BITRATE_KBPS:
            logger.error(
                "Битрейт %.0f Кбит/с ниже минимального (%d Кбит/с): %s",
                bitrate_kbps, MIN_BITRATE_KBPS, path.name,
            )
            raise LowBitrateError(
                f"Битрейт {bitrate_kbps:.0f} Кбит/с ниже минимально допустимого "
                f"({MIN_BITRATE_KBPS} Кбит/с): {path.name}"
            )
        if bitrate_kbps < RECOMMENDED_BITRATE_KBPS:
            logger.warning(
                "Битрейт %.0f Кбит/с ниже рекомендуемого (%d Кбит/с): %s",
                bitrate_kbps, RECOMMENDED_BITRATE_KBPS, path.name,
            )

