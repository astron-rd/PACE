#pragma once

#include <complex>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>

using VisibilityType = std::complex<float>;

#pragma pack(push, 1)

struct UVW {
  float u;
  float v;
  float w;
};

struct Coordinate {
  int x;
  int y;
  int z;
};

struct Metadata {
  int baseline;
  int time_index;
  int nr_timesteps;
  int channel_begin;
  int channel_end;
  Coordinate coordinate;
};

#pragma pack(pop)

constexpr int FourierDomainToImageDomain = 0;
constexpr int ImageDomainToFourierDomain = 1;

struct Inputs {
  xt::xarray<UVW> uvws;
  xt::xarray<float> frequencies;
  xt::xarray<float> wavenumbers;
  xt::xarray<Metadata> metadata;
  xt::xarray<VisibilityType> visibilities;
  xt::xarray<float> taper;

  size_t nr_subgrids;
  float image_size;
};
