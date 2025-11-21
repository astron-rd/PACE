#include <benchmark/benchmark.h>
#include <cstddef>
#include <predict/PointSourceCollection.h>
#include <predict/Predict.h>
#include <predict/PredictPlan.h>
#include <predict/PredictPlanExecCPU.h>

#include <xtensor/xlayout.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include <predict/Directions.h>
#include <predict/test/Common.h>

using predict::PointSourceCollection;
using predict::PredictPlan;
using predict::PredictPlanExecCPU;

const std::vector<int64_t> nstations = {24, 48};
const std::vector<int64_t> nsources = {256, 1024};

class PhasesBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) override {
    n_stations = state.range(0);
    n_channels = state.range(1);
    n_sources = state.range(2);

    PredictPlan plan{};
    plan.nstations = n_stations;
    plan.nchannels = n_channels;
    plan.nbaselines = n_stations * (n_stations - 1) / 2;
    plan.nstokes = 4;
    plan.frequencies.resize({n_channels});
    plan.uvw.resize({n_stations, 3});
    plan.baselines.resize(plan.nbaselines);
    plan.channel_widths = std::vector<double>(n_channels, 1.0e6f);

    plan_exec = std::make_unique<PredictPlanExecCPU>(plan);

    plan_exec->uvw =
        xt::xtensor<double, 2, xt::layout_type::column_major>({n_stations, 3});
    for (size_t i = 0; i < n_stations; ++i) {
      plan_exec->uvw(i, 0) = 100.0 + i;
      plan_exec->uvw(i, 1) = 200.0 + i;
      plan_exec->uvw(i, 2) = 300.0 + i;
    }

    plan_exec->frequencies.resize({n_channels});
    for (size_t i = 0; i < n_channels; ++i) {
      plan_exec->frequencies[i] = 1e8 + i * 1e6;
    }

    plan_exec->lmn =
        xt::random::rand<double>(std::array<size_t, 2>{n_sources, 3}, 0.0, 1.0);

    // Do a single run beforehand to allocate the shift data.
    plan_exec->ComputeStationPhases(true);
  }

  void TearDown(benchmark::State &) override { plan_exec.reset(); }

protected:
  std::unique_ptr<PredictPlanExecCPU> plan_exec;
  size_t n_stations;
  size_t n_channels;
  size_t n_sources;
};

BENCHMARK_DEFINE_F(PhasesBenchmark, ComputePhases)(benchmark::State &state) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  for (auto _ : state) {
    plan_exec->ComputeStationPhases(false);
  }
}

BENCHMARK_REGISTER_F(PhasesBenchmark, ComputePhases)
    ->ArgsProduct({nstations, {64, 128, 256}, nsources})
    ->ArgNames({"#stations", "#channels", "#sources"})
    ->Unit(benchmark::kMillisecond);
