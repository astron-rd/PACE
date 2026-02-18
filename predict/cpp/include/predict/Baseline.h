// Baseline.h: Pair of stations that together form a baseline (interferometer).
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/// \file
/// \brief Pair of stations that together form a baseline (interferometer).

#ifndef PREDICT_BASELINE_H
#define PREDICT_BASELINE_H

#include <cstddef>
#include <utility>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace predict {

typedef std::pair<size_t, size_t> Baseline;

std::vector<Baseline> MakeBaselines(const std::vector<int> &antenna1,
                                    const std::vector<int> &antenna2);
std::vector<int> NSetupSplitUVW(unsigned int nant, const std::vector<int> &ant1,
                                const std::vector<int> &ant2);

std::vector<int>
NSetupSplitUVW(unsigned int nant, const std::vector<int> &antennas1,
               const std::vector<int> &antennas2,
               const xt::xtensor<double, 2> &antenna_positions);

void NSplitUVW(const std::vector<int> &baseline_indices,
               const std::vector<Baseline> &baselines,
               const xt::xtensor<double, 2> &uvw_bl,
               xt::xtensor<double, 2> &uvw_ant);
} // namespace predict

#endif // PREDICT_BASELINE_H
