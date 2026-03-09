#include <cstddef>
#include <iostream>
#include <string>

#include <benchmark/benchmark.h>
#include <predict/test/Common.h>

void ParseCustomFlags(int argc, char **argv) {
  extern bool do_randomized_run;
  extern int randomized_run_seed;

  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == std::string("--randomize")) {
      do_randomized_run = true;
    }

    if (std::string(argv[i]) == "--seed") {
      if (i + 1 < argc) {
        randomized_run_seed = std::stoi(argv[i + 1]);
        ++i;
      } else {
        std::cerr << "Error: --seed option requires an argument." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  std::cout << "Randomized run: " << (do_randomized_run ? "true" : "false");
  if (randomized_run_seed) {
    std::cout << ", seed: " << randomized_run_seed;
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  ParseCustomFlags(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}