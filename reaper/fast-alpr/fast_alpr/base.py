"""
Base module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | list[float]


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        """Perform detection on the input frame and return a list of detections."""


class BaseOCR(ABC):
    @abstractmethod
    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        """Perform OCR on the cropped plate image and return the recognized text and character
        probabilities."""
