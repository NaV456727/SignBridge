from dataclasses import dataclass


@dataclass
class Landmark:
    id: int
    x: float
    y: float
    z: float