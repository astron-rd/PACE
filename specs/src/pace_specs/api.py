from importlib.metadata import metadata

from fastapi import FastAPI

from pace_specs.models import BenchmarkResult

project = metadata("pace-specs")
app = FastAPI(title=project["Name"], version=project["Version"])


@app.post("/results")
def submit(result: BenchmarkResult) -> BenchmarkResult:
    return result
