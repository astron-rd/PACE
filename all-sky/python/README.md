# All-sky Python

## Usage

### Unit tests

```sh
pre-commit run --hook-stage manual --all -v pytest-all-sky
```

### Linting

```sh
pre-commit run --all
```

### Packaging

```sh
pre-commit run --hook-stage manual --all -v build-all-sky
```

### Configuring PMT

```sh
uv venv
source .venv/bin/activate
git clone https://git.astron.nl/RD/pmt.git
cd pmt
mkdir build
cd build
cmake -DPMT_BUILD_PYTHON=on -DPMT_BUILD_RAPL=on -DPMT_BUILD_ROCM=on -DCMAKE_INSTALL_PREFIX=../../.venv/ ..
make
make install
cd ../..
export LD_LIBRARY_PATH=.venv/
...
```

Use rapl as regular user:

`sudo chmod -R a+r /sys/class/powercap/intel-rapl`

Running pytest benchmarks with additional PMT backends:

`uv run pytest --pmt=rapl,cuda,rocm ...`
