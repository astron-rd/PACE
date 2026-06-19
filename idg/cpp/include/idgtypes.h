#pragma once

#include <complex>
#include <cstdint>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <cstdint>

using VisibilityType = std::complex<float>;

#pragma pack(push, 1)

struct UVW {
  float u;
  float v;
  float w;
};

struct Coordinate {
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

struct Metadata {
  uint32_t baseline;
  uint32_t time_index;
  uint32_t nr_timesteps;
  uint32_t channel_begin;
  uint32_t channel_end;
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
