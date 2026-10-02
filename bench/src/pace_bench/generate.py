from pathlib import Path

import jsonschema_markdown
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue
from pydantic_core import CoreSchema

from pace_bench.models import EXAMPLE, BenchmarkResult

OUTPUT = Path("../docs/general/benchmarking/4-specification.md")


class GenerateOrderedJsonSchema(GenerateJsonSchema):
    def generate(
        self, schema: CoreSchema, mode: JsonSchemaMode = "validation"
    ) -> JsonSchemaValue:
        json_schema = super().generate(schema, mode)
        # Pydantic sorts example keys alphabetically, restore definition order
        json_schema["examples"] = [EXAMPLE]
        return json_schema


def main() -> None:
    """Generate Markdown specification from the JSON Schema."""
    schema = BenchmarkResult.model_json_schema(
        schema_generator=GenerateOrderedJsonSchema
    )
    markdown = jsonschema_markdown.generate(
        schema,
        footer=False,
        hide_empty_columns=True,
        examples_format="json",
    )
    OUTPUT.write_text(markdown)
