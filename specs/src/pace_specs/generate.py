from pathlib import Path

import fracturedjson

from pace_specs.models import BenchmarkResult


def main() -> None:
    """Generate JSON Schema file."""
    schema = BenchmarkResult.model_json_schema()
    Path("schema.json").write_text(fracturedjson.dumps(schema))
