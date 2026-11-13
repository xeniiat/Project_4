"""Извлечение метаданных аудиофайлов.

Модуль отвечает только за чтение метаданных (формат, размер,
длительность, частота дискретизации, каналы, битность).
Поддержка WAV реализована через стандартный модуль ``wave``.
Для сжатых форматов (MP3, FLAC, M4A и др.) зарегистрируйте
дополнительный reader-класс в ``MetadataExtractor``.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioMetadata:
    """Метаданные аудиофайла.

    Объект только для чтения: после создания поля изменить нельзя.

    Attributes:
        format: Формат файла (например, ``"WAV"``, ``"MP3"``).
        file_size: Размер файла в байтах.
        duration: Длительность в секундах.
        sample_rate: Частота дискретизации в герцах.
        channels: Количество каналов (1 — моно, 2 — стерео).
        bit_depth: Битность; для сжатых форматов может быть ``None``.
    """

    format: str
    file_size: int
    duration: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int]

    @property
    def bitrate(self) -> float:
        """Приблизительный битрейт в кбит/с.

        Считается как ``размер_в_битах / длительность / 1000``.
        Для сжатых форматов это оценка, для WAV — точное значение.

        Returns:
            Битрейт в кбит/с или ``0.0``, если длительность неизвестна.
        """
        if self.duration <= 0:
            return 0.0
        return (self.file_size * 8) / self.duration / 1000

    @property
    def file_size_mb(self) -> float:
        """Размер файла в мегабайтах."""
        return self.file_size / (1024 * 1024)

    @property
    def channel_name(self) -> str:
        """Человекочитаемое название канальности."""
        if self.channels == 1:
            return "моно"
        if self.channels == 2:
            return "стерео"
        return f"{self.channels} каналов"


class MetadataReader:
    """Базовый интерфейс для чтения метаданных аудиофайла."""

    def can_read(self, path: Path) -> bool:
        """Возвращает ``True``, если reader умеет читать этот файл."""
        raise NotImplementedError

    def read(self, path: Path) -> AudioMetadata:
        """Читает метаданные из файла.

        Raises:
            ValueError: если файл не удаётся разобрать.
        """
        raise NotImplementedError


class WavMetadataReader(MetadataReader):
    """Читает метаданные WAV через стандартный модуль ``wave``."""

    def can_read(self, path: Path) -> bool:
        """WAV-файлы определяются по расширению ``.wav``."""
        return path.suffix.lower() == ".wav"

    def read(self, path: Path) -> AudioMetadata:
        """Читает параметры WAV-файла.

        Args:
            path: Путь к WAV-файлу.

        Returns:
            Заполненный объект ``AudioMetadata``.

        Raises:
            ValueError: если файл повреждён или не является WAV.
            OSError: если файл не удаётся открыть.
        """
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate if rate else 0.0
                return AudioMetadata(
                    format="WAV",
                    file_size=path.stat().st_size,
                    duration=duration,
                    sample_rate=rate,
                    channels=wf.getnchannels(),
                    bit_depth=wf.getsampwidth() * 8,
                )
        except (wave.Error, EOFError) as exc:
            logger.error("Не удалось прочитать WAV %s: %s", path, exc)
            raise ValueError(f"Повреждённый WAV-файл: {path}") from exc


class MetadataExtractor:
    """Фасад: выбирает подходящий reader по расширению файла."""

    def __init__(self, readers: list[MetadataReader] | None = None) -> None:
        """Создаёт экстрактор.

        Args:
            readers: Список readers в порядке приоритета.
                По умолчанию используется только ``WavMetadataReader``.
        """
        self._readers: list[MetadataReader] = readers or [WavMetadataReader()]

    def extract(self, path: str | Path) -> AudioMetadata:
        """Извлекает метаданные аудиофайла.

        Args:
            path: Путь к аудиофайлу.

        Returns:
            Заполненный объект ``AudioMetadata``.

        Raises:
            FileNotFoundError: если файл не существует.
            ValueError: если формат не поддерживается ни одним reader.
        """
        path = Path(path)
        if not path.is_file():
            logger.error("Файл не найден: %s", path)
            raise FileNotFoundError(path)

        for reader in self._readers:
            if reader.can_read(path):
                logger.info("Читаем метаданные %s через %s", path, type(reader).__name__)
                return reader.read(path)

        logger.error("Неподдерживаемый формат: %s", path.suffix)
        raise ValueError(
            f"Формат '{path.suffix}' не поддерживается. "
            f"Доступны: WAV."
        )