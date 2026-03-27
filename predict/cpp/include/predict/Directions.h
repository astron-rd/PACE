// Directions.h: A collection of directions
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief A collection of directions

#ifndef PREDICT_DIRECTIONS_H_
#define PREDICT_DIRECTIONS_H_

#include <span>

#include <predict/Direction.h>
#include <predict/ObjectCollection.h>
#include <predict/common/SmartVector.h>
#ifdef ENABLE_TRACY_PROFILING
#include <tracy/Tracy.hpp>
#endif
#include <vector>
#include <xsimd/xsimd.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace predict {

struct radec2lmn {
  template <class C, class Tag, class Arch>
  void operator()(Arch, const predict::Direction &reference, const C &ra,
                  const C &dec, xt::xtensor<double, 2> &lmn, Tag);
};

template <class C, class Tag, class Arch>
void radec2lmn::operator()(Arch, const predict::Direction &reference,
                           const C &ra, const C &dec,
                           xt::xtensor<double, 2> &lmn, Tag) {
  using b_type = xsimd::batch<double, Arch>;
  std::size_t inc = b_type::size;
  std::size_t size = ra.size();
  double sin_dec0 = std::sin(reference.dec);
  double cos_dec0 = std::cos(reference.dec);
  // size for which the vectorization is possible
  std::size_t vec_size = size - size % inc;

  for (std::size_t i = 0; i < vec_size; i += inc) {
    const b_type ra_vec = b_type::load(&ra[i], Tag());
    const b_type dec_vec = b_type::load(&dec[i], Tag());

    const b_type delta_ra = ra_vec - reference.ra;

    const std::pair<b_type, b_type> sin_cos_delta_ra = xsimd::sincos(delta_ra);
    const std::pair<b_type, b_type> sin_cos_dec = xsimd::sincos(dec_vec);

    const b_type &cos_dec = sin_cos_dec.second;
    const b_type &sin_dec = sin_cos_dec.first;
    const b_type &sin_delta_ra = sin_cos_delta_ra.first;
    const b_type &cos_delta_ra = sin_cos_delta_ra.second;

    const b_type result_vec_l = cos_dec * sin_delta_ra;
    const b_type result_vec_m =
        sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra;
    const b_type result_vec_n =
        sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra;

    xsimd::store(&lmn(0, i), result_vec_l, Tag());
    xsimd::store(&lmn(1, i), result_vec_m, Tag());
    xsimd::store(&lmn(2, i), result_vec_n, Tag());
  }
  // Remaining part that cannot be vectorize
  for (std::size_t i = vec_size; i < size; ++i) {
    double delta_ra = ra[i] - reference.ra;
    double sin_delta_ra = std::sin(delta_ra);
    double cos_delta_ra = std::cos(delta_ra);
    double sin_dec = std::sin(dec[i]);
    double cos_dec = std::cos(dec[i]);

    lmn(0, i) = cos_dec * sin_delta_ra;
    lmn(1, i) = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra;
    lmn(2, i) = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra;
  }
}
enum computation_strategy { XSIMD };

class Directions : ObjectCollection<Direction> {
public:
  Directions() : ra(), dec() {}
  void Add(const Direction &direction) {
    ra.push_back(direction.ra);
    dec.push_back(direction.dec);
  }

  void Add(const std::span<Direction> &directions) {
    ra.reserve(Size() + directions.size());
    dec.reserve(Size() + directions.size());
    for (size_t i = 0; i < directions.size(); ++i) {
      ra.push_back(directions[i].ra);
      dec.push_back(directions[i].dec);
    }
  }

  void Reserve(size_t new_size) {
    ra.reserve(new_size);
    dec.reserve(new_size);
  }

  void Resize(size_t new_size) {
    ra.resize(new_size);
    dec.resize(new_size);
  }

  void Clear() {
    ra.clear();
    dec.clear();
  }

  Direction operator[](size_t i) const { return Direction(ra[i], dec[i]); }

  size_t Size() const { return ra.size(); }

  SmartVector<double> ra;
  SmartVector<double> dec;

  /**
   * Compute LMN coordinates of \p direction relative to \p reference.
   * \param[in]   reference
   * Reference direction on the celestial sphere.
   * \param[in]   direction
   * Direction of interest on the celestial sphere.
   * \param[out]   lmn
   * Pointer to a buffer of (at least) length three into which the computed LMN
   * coordinates will be written.
   */
  template <computation_strategy T = computation_strategy::XSIMD>
  inline void RaDec2Lmn(const Direction &reference,
                        xt::xtensor<double, 2> &lmn) const;
};

template <>
inline void Directions::RaDec2Lmn<Directions::computation_strategy::MULTI>(
    const Direction &reference, xt::xtensor<double, 2> &lmn) const {
  /**
   * \f{eqnarray*}{
   *   \ell &= \cos(\delta) \sin(\alpha - \alpha_0) \\
   *      m &= \sin(\delta) \cos(\delta_0) - \cos(\delta) \sin(\delta_0)
   *                                         \cos(\alpha - \alpha_0)
   * \f}
   */

  const auto ra_view = ra.view();
  const auto dec_view = dec.view();

  const xt::xtensor<double, 1> delta_ra = ra_view - reference.ra;
  const xt::xtensor<double, 1> sin_delta_ra = xt::sin(delta_ra);
  const xt::xtensor<double, 1> cos_delta_ra = xt::cos(delta_ra);
  const xt::xtensor<double, 1> sin_dec = xt::sin(dec_view);
  const xt::xtensor<double, 1> cos_dec = xt::cos(dec_view);
  const double sin_dec0 = std::sin(reference.dec);
  const double cos_dec0 = std::cos(reference.dec);

  xt::view(lmn, xt::all(), 0) = cos_dec * sin_delta_ra;
  xt::view(lmn, xt::all(), 1) =
      sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra;
  // Normally, n is calculated using n = std::sqrt(1.0 - l * l - m * m).
  // However the sign of n is lost that way, so a calculation is used that
  // avoids losing the sign. This formula can be found in Perley (1989).
  // Be aware that we asserted that the sign is wrong in Perley (1989),
  // so this is corrected.
  xt::view(lmn, xt::all(), 2) =
      sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra;
}

template <>
inline void Directions::RaDec2Lmn<Directions::computation_strategy::SINGLE>(
    const Direction &reference, xt::xtensor<double, 2> &lmn) const {
  /**
   * \f{eqnarray*}{
   *   \ell &= \cos(\delta) \sin(\alpha - \alpha_0) \\
   *      m &= \sin(\delta) \cos(\delta_0) - \cos(\delta) \sin(\delta_0)
   *                                         \cos(\alpha - \alpha_0)
   * \f}
   */
  for (size_t i = 0; i < ra.size(); i++) {
    const double delta_ra = ra[i] - reference.ra;
    const double sin_delta_ra = std::sin(delta_ra);
    const double cos_delta_ra = std::cos(delta_ra);
    const double sin_dec = std::sin(dec[i]);
    const double cos_dec = std::cos(dec[i]);
    const double sin_dec0 = std::sin(reference.dec);
    const double cos_dec0 = std::cos(reference.dec);

    lmn(i, 0) = cos_dec * sin_delta_ra;
    lmn(i, 1) = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra;
    // Normally, n is calculated using n = std::sqrt(1.0 - l * l - m * m).
    // However the sign of n is lost that way, so a calculation is used that
    // avoids losing the sign. This formula can be found in Perley (1989).
    // Be aware that we asserted that the sign is wrong in Perley (1989),
    // so this is corrected.
    lmn(i, 2) = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra;
  }
}

template <>
inline void Directions::RaDec2Lmn<Directions::computation_strategy::XSIMD>(
    const Direction &reference, xt::xtensor<double, 2> &lmn) const {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  xt::xtensor<double, 2> lmn_tmp({3, ra.size()});
  xsimd::dispatch(radec2lmn{})(reference, ra, dec, lmn_tmp,
                               xsimd::unaligned_mode());
  lmn = xt::transpose(lmn_tmp);
}

} // namespace predict

#endif