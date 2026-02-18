#ifndef PREDICT_BENCHMARK_COMMON_HPP
#define PREDICT_BENCHMARK_COMMON_HPP

#include <predict/Directions.h>
#include <predict/GaussianSourceCollection.h>
#include <predict/PointSourceCollection.h>
#include <predict/PredictPlanExecCPU.h>
#include <vector>

#include <xtensor/xtensor.hpp>

#include <chrono>
#include <predict/Baseline.h>
#include <predict/GaussianSource.h>
#include <predict/PointSource.h>
#include <predict/Predict.h>
#include <predict/Stokes.h>
#ifdef ENABLE_TRACY_PROFILING
#include <tracy/Tracy.hpp>
#endif
extern bool do_randomized_run;

namespace utils {
class SimpleScopedTimer {
public:
  SimpleScopedTimer(std::string name) {
    timer_name = name;
    std::cout << "Started: " << name << std::endl;
    start_ = std::chrono::steady_clock::now();
  }

  ~SimpleScopedTimer() {
    std::chrono::time_point<std::chrono::steady_clock> end =
        std::chrono::steady_clock::now();

    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);

    std::cout << timer_name << ": elapsed " << duration_ms.count() << " ms"
              << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::string timer_name;
};

class WeightedScopedTimer {
public:
  WeightedScopedTimer(std::string name, double weight) {
    timer_name_ = name;
    weight_ = weight;
    std::cout << "Started: " << name << std::endl;
    start_ = std::chrono::steady_clock::now();
  }
  ~WeightedScopedTimer() {
    auto end = std::chrono::steady_clock::now();

    double duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count();

#ifdef ENABLE_TRACY_PROFILING
    const std::string timer_description = timer_name_ + " elapsed:";
    const std::string timer_avg_description = timer_name_ + "avg:";
    TracyPlot(timer_description.c_str(), duration_ms);
    TracyPlot(timer_avg_description.c_str(), duration_ms / weight_);
#endif

    std::cout << timer_name_ << ": elapsed " << duration_ms << " ms, approx "
              << (duration_ms / weight_) << " ms per unit" << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::string timer_name_;
  double weight_;
};

} // namespace utils

namespace predict {

struct PredictRun {
  PredictRun(size_t nstokes, size_t nchannels, size_t nstations,
             size_t nbaselines) {
    plan.nstokes = nstokes;
    plan.nchannels = nchannels;
    plan.nstations = nstations;
    plan.nbaselines = nbaselines;
  }

  void Initialize() {
    for (size_t st1 = 0; st1 < plan.nstations - 1; ++st1) {
      for (size_t st2 = st1 + 1; st2 < plan.nstations; ++st2) {
        plan.baselines.emplace_back(Baseline(st1, st2));
      }
    }

    plan.frequencies.resize({plan.nchannels});

    for (size_t chan = 0; chan < plan.nchannels; ++chan) {
      plan.frequencies(chan) = 130.0e6 + chan * 1.0e6;
    }

    plan.uvw.resize({plan.nstations, 3});
    for (size_t st = 0; st < plan.nstations; ++st) {
      plan.uvw(st, 0) = st * 5000;
      plan.uvw(st, 1) = st * 1000;
      plan.uvw(st, 2) = 0;
    }

    plan.channel_widths = std::vector<double>(plan.nchannels, 1.0e6f);

    plan.baselines.resize(plan.nbaselines);
    Clean();
  }

  void Run(PointSourceCollection &sources) {
    PredictPlanExecCPU plan_exec{plan};
    Predict pred;
    sources.EvaluateSpectra(plan.frequencies);
    pred.run(plan_exec, sources, buffer);
  }

  void Run(GaussianSourceCollection &sources) {
    PredictPlanExecCPU plan_exec{plan};
    Predict pred;
    sources.EvaluateSpectra(plan.frequencies);
    pred.run(plan_exec, sources, buffer);
  }

  void RunWithStrategy(PointSourceCollection &sources,
                       const computation_strategy strat) {
    PredictPlanExecCPU plan_exec{plan};
    Predict pred;
    sources.EvaluateSpectra(plan.frequencies);
    pred.runWithStrategy(plan_exec, sources, buffer, strat);
  }

  void RunWithStrategy(GaussianSourceCollection &sources,
                       const computation_strategy strat) {
    PredictPlanExecCPU plan_exec{plan};
    Predict pred;
    sources.EvaluateSpectra(plan.frequencies);
    pred.runWithStrategy(plan_exec, sources, buffer, strat);
  }

  void Clean() {
    buffer = Buffer4D({plan.nstokes, plan.nbaselines, 2, plan.nchannels}, 0.0);
  }

  template <typename SourceType> SourceType makeSource() const {
    return SourceType{offset_source, Stokes{1.0, 0.0, 0.0, 0.0}};
  }

  Buffer4D buffer;

  Direction reference_point;
  Direction offset_source;
  bool stokes_i_only = false;
  bool correct_freq_smearing = false;
  PredictPlan plan;
};

std::unique_ptr<PredictRun> MakePredictRun(const Direction &reference_point,
                                           const Direction &offset_point,
                                           size_t n_stations, size_t n_channels,
                                           bool stokes_i_only,
                                           bool correct_freq_smearing);
void SetUpRandomizedSource(Direction &test_reference_point,
                           Direction &test_offset_point, int seed);

void SetUpFixedSource(Direction &test_reference_point,
                      Direction &test_offset_point);

} // namespace predict

inline float ComputeSmearterm(double uvw, double halfwidth) {
  float smearterm = static_cast<float>(uvw) * static_cast<float>(halfwidth);

  return (smearterm == 0.0f) ? 1.0f
                             : std::fabs(std::sin(smearterm) / smearterm);
}
#endif // PREDICT_BENCHMARK_COMMON_HPP