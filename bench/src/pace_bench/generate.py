from pathlib import Path

import jsonschema_markdown
from loguru import logger
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue
from pydantic_core import CoreSchema

from pace_bench.models import EXAMPLE, Result

OUTPUT = Path("../docs/general/benchmarking/4-specification.md")
DEFS_PREFIX = "#/$defs/"


def referenced_defs(value: object, defs: JsonSchemaValue, found: list[str]) -> None:
    """Collect definition names in depth-first order of first reference."""
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith(DEFS_PREFIX):
            name = ref.removeprefix(DEFS_PREFIX)
            if name not in found:
                found.append(name)
                referenced_defs(defs[name], defs, found)
        for item in value.values():
            referenced_defs(item, defs, found)
    elif isinstance(value, list):
        for item in value:
            referenced_defs(item, defs, found)


class GenerateOrderedJsonSchema(GenerateJsonSchema):
    def generate(
        self, schema: CoreSchema, mode: JsonSchemaMode = "validation"
    ) -> JsonSchemaValue:
        json_schema = super().generate(schema, mode)
        # Undo automatic key sorting by Pydantic
        json_schema["examples"] = [EXAMPLE]
        defs = json_schema.get("$defs", {})
        names: list[str] = []
        referenced_defs(json_schema["properties"], defs, names)
        json_schema["$defs"] = {name: defs[name] for name in names}
        return json_schema


def main() -> None:
    """Generate Markdown specification from the JSON Schema."""
    schema = Result.model_json_schema(schema_generator=GenerateOrderedJsonSchema)
    # Silence log output from jsonschema-markdown
    logger.disable("jsonschema_markdown")
    markdown = jsonschema_markdown.generate(
        schema,
        footer=False,
        hide_empty_columns=True,
        examples_format="json",
    )
    OUTPUT.write_text(markdown)
