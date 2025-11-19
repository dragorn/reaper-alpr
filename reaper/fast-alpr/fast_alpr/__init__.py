"""
FastALPR package.
"""

from fast_alpr.alpr import ALPR, ALPRResult
from fast_alpr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult

__all__ = ["ALPR", "ALPRResult", "BaseDetector", "BaseOCR", "DetectionResult", "OcrResult"]
