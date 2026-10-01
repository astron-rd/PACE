# Specifications

The OpenAPI specifications for PACE, defined with Pydantic and FastAPI.
The generated specifications are in [`openapi.json`](openapi.json).

## Updating the specifications

1. Edit or add a model in [`src/pace_specs/models.py`](src/pace_specs/models.py) and use it in a route in [`src/pace_specs/api.py`](src/pace_specs/api.py). See the [Pydantic documentation on models](https://pydantic.dev/docs/validation/latest/concepts/models/) for the available field types and constraints, and the [FastAPI documentation on request bodies](https://fastapi.tiangolo.com/tutorial/body/) for using models in routes.

1. Preview the changes in the interactive documentation at <http://127.0.0.1:8000/docs>:

   ```sh
   uv run fastapi dev
   ```

1. Bump the specifications version, using `major`, `minor` or `patch` depending on the change:

   ```sh
   uv version --bump minor
   ```

1. Update `openapi.json`:

   ```sh
   uv run export
   ```

1. Commit the changes and open a pull request.
