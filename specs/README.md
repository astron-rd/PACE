# Specifications

The JSON Schema specifications for PACE, defined with Pydantic.
The generated schema is in [`schema.json`](schema.json).

## Updating the specifications

1. Edit the model in [`src/pace_specs/models.py`](src/pace_specs/models.py). See the [Pydantic documentation on models](https://pydantic.dev/docs/validation/latest/concepts/models/) for the available field types and constraints.

1. Update `schema.json`:

   ```sh
   uv run export
   ```

1. Commit the changes and open a pull request.
