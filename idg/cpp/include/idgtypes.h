#pragma once

#include <complex>

using VisibilityType = std::complex<float>;
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

constexpr int FourierDomainToImageDomain = 0;
constexpr int ImageDomainToFourierDomain = 1;