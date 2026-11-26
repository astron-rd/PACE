import argparse
import json
from datetime import datetime
from importlib import metadata
from pathlib import Path

from pace_bench import hardware
from pace_bench.models import Result


def main() -> None:
    """Record the parameters and timings of a run as a result file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "measurements", type=Path, help="JSON file with parameters and timings"
    )
    parser.add_argument("--application", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--framework")
    parser.add_argument(
        "--commit", required=True, help="git commit hash of the benchmarked code"
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        type=datetime.fromisoformat,
        help="benchmark start time in ISO 8601 format with UTC offset",
    )
    args = parser.parse_args()

    measurements = json.loads(args.measurements.read_text())
    result = Result.model_validate(
        {
            **measurements,
            "version": metadata.version("pace-bench"),
            "timestamp": args.timestamp,
            "application": args.application,
            "language": args.language,
            "framework": args.framework,
            "commit": args.commit[:8],
            "hardware": hardware.detect(),
        }
    )

    name = f"{int(result.timestamp.timestamp())}-{result.framework or 'native'}.json"
    path = Path("results", result.application, result.language, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2, exclude_none=True) + "\n")
    print(path)
