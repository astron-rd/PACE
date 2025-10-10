// SkyModel.h: Utility to read a sky model text into SourceCollections
// rotation measure.
//
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache-2.0

#ifndef SKY_MODEL_H_
#define SKY_MODEL_H_

#include "GaussianSourceCollection.h"
#include "PointSourceCollection.h"

namespace predict {
void ParseSkyModel(const std::string &skymodel_path,
                   GaussianSourceCollection &sources,
                   PointSourceCollection &point_sources);
}
#endif // SKY_MODEL_H_