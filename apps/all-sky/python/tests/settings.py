from dataclasses import dataclass

from tests.measurements.data import Settings


@dataclass
class AllSkySettings(Settings):
    image_size_x: int = 256
    image_size_y: int = 256
    frequency: float = 58593750.0
    test_data_path: str = "tests.data/data.h5"


# Benchmark repeated image generation on same visibilities
BENCH_SETTINGS_SINGLE = AllSkySettings(
    image_size_x=256,
    image_size_y=256,
    warmup=30,
    iterations=30,
    variances=1,
    name="REPLACE ME",
)

# Benchmark image generation across different visibilities
BENCH_SETTINGS_MANY = AllSkySettings(
    image_size_x=256,
    image_size_y=256,
    warmup=30,
    iterations=30,
    variances=10,
    name="REPLACE ME",
)
