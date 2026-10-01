import json

from pace_specs.api import app


def main() -> None:
    """Generate OpenAPI schema file."""
    with open("openapi.json", "w") as file:
        json.dump(app.openapi(), file, indent=2)
        file.write("\n")
