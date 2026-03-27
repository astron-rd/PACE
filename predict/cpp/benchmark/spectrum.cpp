#include <benchmark/benchmark.h>
#include <predict/Predict.h>
#include <random>
#include <xtensor/xtensor.hpp>

#include <predict/Direction.h>
#include <predict/PointSource.h>
#include <predict/Spectrum.h>

const std::vector<int64_t> n_frequencies = {128, 256, 512};

const double kHz = 1.e3;
const double MHz = 1.e6;
namespace {} // namespace
using namespace predict;
class SpectrumBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) override {
    n_frequencies = state.range(0);
    const bool is_log = state.range(1);
    const bool compute_rotation = state.range(2);
    frequencies.resize({n_frequencies});
    for (int64_t i = 0; i < n_frequencies; i++) {
      frequencies(i) = static_cast<double>(i) * 100. * kHz + MHz;
    }

    spectrum = std::make_unique<Spectrum>();
    auto &spec = *spectrum;

    spec.SetSpectralTerms(MHz, is_log,
                          xt::xtensor<double, 1>({1.0e-6, 2.0e-6, 3.0e-6}));
    spec.SetReferenceFlux({1.0, 0.0, 0.0, 0.0});
    spec.SetRotationMeasure(0.1, compute_rotation);
    spec.SetPolarization(0.3, 0.2);

    computed_spectrum = std::make_unique<StokesVector>();

    point_source = std::make_unique<predict::PointSource>(
        predict::Direction(0.0, 0.0), spec.GetReferenceFlux());

    point_source->SetSpectralTerms(
        spec.GetReferenceFrequency(), spec.HasLogarithmicSpectralIndex(),
        spec.GetSpectralTerms().begin(), spec.GetSpectralTerms().end());
    if (compute_rotation) {
      point_source->SetRotationMeasure(spec.GetPolarizationFactor(),
                                       spec.GetPolarizationAngle(),
                                       spec.GetRotationMeasure());
    }
  }
  void TearDown(benchmark::State &) override { spectrum.reset(); }

protected:
  std::unique_ptr<Spectrum> spectrum;
  std::unique_ptr<StokesVector> computed_spectrum;
  size_t n_frequencies;
  xt::xtensor<double, 1> frequencies;
  std::unique_ptr<predict::PointSource> point_source;
};

BENCHMARK_DEFINE_F(SpectrumBenchmark, SpectrumBenchmark)
(benchmark::State &state) {
  for (auto _ : state) {
    spectrum->Evaluate(frequencies, *computed_spectrum);
  }
}

BENCHMARK_DEFINE_F(SpectrumBenchmark, SpectrumReference)
(benchmark::State &state) {
  for (auto _ : state) {
    for (int64_t i = 0; i < n_frequencies; i++) {
      benchmark::DoNotOptimize(point_source->GetStokes(frequencies(i)).I);
    }
  }
}

BENCHMARK_REGISTER_F(SpectrumBenchmark, SpectrumBenchmark)
    ->ArgsProduct({
        n_frequencies,
        {false, true},
        {false, true},

    })
    ->ArgNames({"#frequencies", "is_log", "compute_rotation"})
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(SpectrumBenchmark, SpectrumReference)
    ->ArgsProduct({
        n_frequencies,
        {false, true},
        {false, true},

    })
    ->ArgNames({"#frequencies", "is_log", "compute_rotation"})
    ->Unit(benchmark::kNanosecond);
