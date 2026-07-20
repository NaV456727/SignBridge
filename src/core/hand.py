from dataclasses import dataclass

from src.core.landmark import Landmark

@dataclass
class Hand:
    landmarks: list[Landmark]
    bbox: tuple
    handedness: str
    confidence: float