"""
Извлечение метаданных аудиофайлов.
Модуль читает заголовочную информацию (размер, длительность, частоту
дискретизации, битность) из файла, который уже прошёл проверку
требований.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tinytag import TinyTag

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioMetadata:
    """Метаданные аудиофайла.

    Объект только для чтения: после создания поля изменить нельзя.

    Attributes:
        file_size: Размер файла в байтах.
        duration: Длительность в секундах.
        sample_rate: Частота дискретизации в герцах.
        bit_depth: Битность; для сжатых форматов может быть ``None``.
    """

    file_size: int
    duration: float
    sample_rate: int
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


class TinyTagMetadataReader(MetadataReader):
    """Читает метаданные через ``tinytag`` (MP3, FLAC, M4A, WAV, OGG)."""

    def can_read(self, path: Path) -> bool:
        """``tinytag`` поддерживает все форматы из требований площадки."""
        return True  # файл уже прошёл проверку в loader

    def read(self, path: Path) -> AudioMetadata:
        """Читает параметры аудиофайла.

        Args:
            path: Путь к аудиофайлу (уже проверенному).

        Returns:
            Заполненный объект ``AudioMetadata``.

        Raises:
            ValueError: если файл не удаётся разобрать.
        """
        try:
            tag = TinyTag.get(str(path))
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать метаданные: {path}") from exc

        return AudioMetadata(
            file_size=path.stat().st_size,
            duration=float(tag.duration or 0.0),
            sample_rate=int(tag.samplerate or 0),
            bit_depth=getattr(tag, "bitdepth", None),
        )


class MetadataExtractor:
    """Выбирает подходящий reader по расширению файла."""

    def __init__(self, readers: list[MetadataReader] | None = None) -> None:
        """Создаёт экстрактор.

        Args:
            readers: Список readers в порядке приоритета.
                По умолчанию — ``TinyTagMetadataReader``.
        """
        if readers is None:
            readers = [TinyTagMetadataReader()]
        self._readers = readers

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
            f"Формат '{path.suffix}' не поддерживается."
        )