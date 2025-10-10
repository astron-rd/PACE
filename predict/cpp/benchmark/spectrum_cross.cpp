#include <benchmark/benchmark.h>
#include <xtensor/xtensor.hpp>

#include <predict/Direction.h>
#include <predict/PointSource.h>
#include <predict/Spectrum.h>
const std::vector<int64_t> n_frequencies = {128, 256, 512};

const double kHz = 1.e3;
const double MHz = 1.e6;
namespace {} // namespace
using namespace predict;
class SpectrumCrossBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) override {
    n_frequencies = state.range(0);
    const bool is_log = state.range(1);
    const bool compute_rotation = state.range(2);
    frequencies.resize({n_frequencies});
    for (int64_t i = 0; i < n_frequencies; i++) {
      frequencies(i) = i * 100. * kHz + MHz;
    }

    spectrum = std::make_unique<Spectrum>();
    auto &spec = *spectrum;

    spec.SetSpectralTerms(MHz, is_log,
                          xt::xtensor<double, 1>({1.0e-6, 2.0e-6, 3.0e-6}));
    spec.SetRotationMeasure(0.1);
    spec.SetReferenceFlux({1.0, 0.0, 0.0, 0.0});
    spec.SetPolarization(0.3, 0.2);

    computed_spectrum = std::make_unique<xt::xtensor<double, 3>>();
    computed_spectrum->resize({2, 4, n_frequencies});
  }
  void TearDown(benchmark::State &) override {
    spectrum.reset();
    computed_spectrum.reset();
  }

protected:
  std::unique_ptr<Spectrum> spectrum;
  std::unique_ptr<xt::xtensor<double, 3>> computed_spectrum;
  size_t n_frequencies;
  xt::xtensor<double, 1> frequencies;
};

BENCHMARK_DEFINE_F(SpectrumCrossBenchmark, SpectrumBenchmark)
(benchmark::State &state) {
  for (auto _ : state) {
    spectrum->EvaluateCrossCorrelations(frequencies, *computed_spectrum);
  }
}

BENCHMARK_REGISTER_F(SpectrumCrossBenchmark, SpectrumBenchmark)
    ->ArgsProduct({
        n_frequencies,
        {false, true},
        {false, true},

    })
    ->ArgNames({"#frequencies", "is_log", "compute_rotation"})
    ->Unit(benchmark::kNanosecond);
