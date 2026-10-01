from typing import Literal

from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    application: Literal["all-sky", "dedisp", "idg"]
    language: Literal["python", "rust", "julia"]
    commit: str = Field(min_length=7, max_length=40)
    timings: dict[str, float]
