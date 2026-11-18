"""Loading audio files into a numpy array."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


class AudioLoadError(Exception):
    """Base error for audio loading."""


class AudioFileNotFoundError(AudioLoadError):
    """Файл не найден."""


class UnsupportedFormatError(AudioLoadError):
    """File format is not supported."""


class CorruptedAudioError(AudioLoadError):
    """The file is corrupted, empty, or cannot be read as audio."""


class AudioData:
    """Load audio.
    samples: float32 array of shape (frames, channels), values are in [-1, 1].
    """

    def __init__(self, samples: np.ndarray, sample_rate: int, path: Path) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.path = path
        self.channels: int = samples.shape[1]
        self.duration_sec: float = samples.shape[0] / sample_rate


class AudioLoader:
    """Validate the audio file and load it into memory."""

    def __init__(self, supported_extensions: set[str] = SUPPORTED_EXTENSIONS) -> None:
        self._supported = supported_extensions

    def load(self, path: str | Path) -> AudioData:
        path = Path(path)
        self._validate(path)

        try:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except sf.SoundFileError as exc:
            logger.error("Could not be read %s: %s", path.name, exc)
            raise CorruptedAudioError(
                f"The file is corrupted or is not an audio: {path.name}"
            ) from exc

        if samples.shape[0] == 0:
            raise CorruptedAudioError(f"The file has no audio data: {path.name}")

        audio = AudioData(samples, sample_rate, path)
        logger.info(
            "Loaded %s: %.1f s, %d Hz, channels: %d",
            path.name,
            audio.duration_sec,
            audio.sample_rate,
            audio.channels,
        )
        return audio

    def _validate(self, path: Path) -> None:
        if not path.is_file():
            raise AudioFileNotFoundError(f"The file is not found: {path}")
        if path.suffix.lower() not in self._supported:
            raise UnsupportedFormatError(
                f"Format {path.suffix!r} is not supported. "
                f"Available: {', '.join(sorted(self._supported))}")