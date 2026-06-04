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

### How to use `tox`

Our current ecosystem still relies on `tox` to do the job of `uv`. It creates
the virtualenvs and installs the dependencies to run tests, linting etc. You can
recognize the jobs and steps from the `pyproject.toml` file.

To run tasks you would do: `tox -e py311` to run the unit tests for Python3.11

Other tasks include

1. Linting `tox -e lint`, `tox -e format`, `tox -e fix`
1. Code coverage `tox -e coverage`
1. Unit tests, different python version `tox -e py313`, `tox -e py314`

### What do I need to do coming from `pre-commit`

Just be sure to install `tox>4.0`, The `pre-commit` will just call the
individual `tox` tasks and bobs our uncle.

### What else is out there?

1. poetry
1. hatch
1. rye
1. PDM

But stick for `uv` it has pretty much won this fight and would be a good
investment to learn. 2nd place to poetry.
