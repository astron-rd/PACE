Python is an interpreted programming language know for its ease of use and low
barrier to entree.

## Automation & Frameworks

While Python's language philosophy is that there should only be one way to do
something, ironically, its tools, automation and frameworks are extremely
fragmented.

Here is a TL;DR:

Python, comes without the following tools:

1. Packaging framework (Think; deb, rpm, cargo, CPM, vcpkg)
1. Packaging tool (Think; CMake)
1. Testing (Think; Catch2, ctest)
1. Linting (Think; cmake-format)

Because of these, everyone under the sun has decided that for each of these
categories they should make such a tool, hence this ecosystem is highly
fragmented.

### History crash course

1. Because many Python libraries are incompatible with one another, it is
   standard practice to install tools in so called `virtual environments`
   (virtualenv). Historically these where created by using the `virtualenv`
   package (library).

1. To install packages, `pip` is used, this tool downloads packages from
   pypi.org.

1. To make your python project something you can install you use the
   `setuptools` package. This is the _packaging tool_. However, historically,
   this tool required writing a `setup.py` script which meant you could do
   anything, this lead to a lot of none-portability problems.

1. To solve the issue with 3. we moved to `setup.cfg`. However, this is specific
   to `setuptools` and there are other packaging libraries.

1. To solve the issue with 4. we moved to `pyproject.toml` this file is coverned
   by a PEP and so its officially native Python. This file can recognize
   different packaging frameworks and build tools all within a single file.

1. But this wasn't enough because `pip` and `virtualenv` are very slow and
   having to combine the two in some glue scripts to integrate with CI/CD or
   `pre-commit` is error prone.

1. So now all the world uses `uv` which uses hardlinks to only download a
   package once and then cache it in `~/.cache/uv`. So `uv` replaces both `pip`
   and `virtualenv`

### Usage of `tox`

Our current ecosystem still relies on `tox` to do the job of `uv`. It creates
the virtualenvs and installs the dependencies to run tests, linting etc.

### Adopting `uv`

The Python ecosystem has settled on `uv` as the modern all-in-one tool for
Python development. For this project `uv` is adopted as the primary
tool for managing dependencies and running tests. In addition to `uv` and its
build tool [`uv_build`](https://docs.astral.sh/uv/concepts/build-backend/), we
are also using [`ruff`](https://docs.astral.sh/ruff/) for linting and
formatting.

This means we are able to launch applications directly as
[scripts](https://docs.astral.sh/uv/#scripts) using `uv run <command>`, and call
[tools](https://docs.astral.sh/uv/#tools) from the CLI using
`uvx <tool> <command>`. All without having to manually manage virtual
environments or dependency installs.

For example, `pre-commit` can be called using `uvx pre-commit run --all-files`.
