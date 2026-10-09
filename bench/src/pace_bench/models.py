from importlib import metadata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat
from pydantic.config import JsonDict
from pydantic_extra_types.semantic_version import SemanticVersion

EXAMPLE: JsonDict = {
    "version": metadata.version("pace-bench"),
    "application": "dedisp",
    "language": "rust",
    "framework": "rayon",
    "commit": "a5298758",
    "hardware": {
        "cpu": {"model": "AMD EPYC 7763", "cores": 64},
        "gpu": {"model": "NVIDIA A100 80GB"},
        "memory": {"gb": 512, "mts": 3200},
    },
    "timings": {
        "input": 0.42,
        "plan": 0.08,
        "preprocessing": 0.15,
        "execution": 3.71,
    },
}


class Cpu(BaseModel):
    """Processor of the machine that ran the benchmark."""

    model: str = Field(description="CPU model name.")
    cores: int = Field(gt=0, description="Number of physical cores.")


class Gpu(BaseModel):
    """Graphics card of the machine that ran the benchmark."""

    model: str = Field(description="GPU model name.")


class Memory(BaseModel):
    """Main memory of the machine that ran the benchmark."""

    gb: int = Field(gt=0, description="Memory size in gigabytes.")
    mts: int = Field(gt=0, description="Memory speed in megatransfers per second.")


class Hardware(BaseModel):
    """Machine that ran the benchmark."""

    cpu: Cpu = Field(description="CPU configuration.")
    gpu: Gpu | None = Field(default=None, description="GPU configuration.")
    memory: Memory = Field(description="Memory configuration.")


class Result(BaseModel):
    """Result of a single benchmark run."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE]})

    version: SemanticVersion = Field(description="Specification SemVer.")
    application: Literal["all-sky", "dedisp", "idg"] = Field(
        description="Benchmarked application."
    )
    language: Literal["cpp", "julia", "python", "rust"] = Field(
        description="Programming language."
    )
    framework: str | None = Field(
        default=None,
        min_length=1,
        description="Acceleration framework.",
        examples=["openmp", "cuda", "rayon", "numba"],
    )
    commit: str = Field(
        pattern=r"^[0-9a-f]{8}$",
        description="Git commit hash.",
    )
    hardware: Hardware = Field(description="Hardware configuration.")
    timings: dict[str, NonNegativeFloat] = Field(
        description="Measurements in seconds per benchmark phase."
    )
