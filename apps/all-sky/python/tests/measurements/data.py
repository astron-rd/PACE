from dataclasses import dataclass


@dataclass
class MeasureData:
    seconds: float
    joules: float
    watts: float


@dataclass
class ResultData:
    min: float = float("inf")
    max: float = float("-inf")
    mean: float = 0.0
    std: float = 0.0


@dataclass
class Settings:
    warmup: int = 30
    iterations: int = 3
    variances: int = 10
    name: str = "NOT DEFINED!"
