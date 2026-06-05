# Rust IDG implementation

This is a Rust implementation of IDG. It can generate input data or read it from an input file, and use Image Domain Gridding to create an image.

## Building

To build the application, you will need Rust. You can install Rust from [rustup](https://rustup.rs).

1. Install the HDF5 and FFTW libraries on your system. You can use the system package manager or the module system if you're building on an HPC cluster.
2. Run `cargo build --release` to build idg.

Your executable will be `target/release/idg[.exe]`.

## Usage

### Generating input

To generate input and then produce an image, use `idg generate`. There are many parameters to customize your image, see them using `idg help generate`.

By default, IDG will only output the finished image. To also output the subgrids, pass the `--output-subgrids` option *before* the `generate` command.

### Loading input

To load input from a file and produce an image, use `idg load`. See the command line options using `idg help load`. To get an input file you can generate one using the input generator in `idg/input`. You can also run `idg generate --output-input` to generate input using the Rust version and create an input file.

By default, IDG will only output the finished image. To also output the subgrids, pass the `--output-subgrids` option *before* the `load` command.
