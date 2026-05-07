# Julia IDG

## How to run

1. Install julia
2. Place `input.h5` in the working directory
3. Run IDG with the following command:

```
julia --project=. -t auto main.jl
```

`--project=.` selects the current directory as the environment for the interpreter. `-t auto` tells julia to spawn the same number of threads as CPU cores which are available.
