// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PREDICT_DIRECTION_H_
#define PREDICT_DIRECTION_H_

#include <cmath>
#include <cstring>
#include <iostream>

namespace predict {

/// \brief A direction on the celestial sphere.
/// @{

struct Direction {
  constexpr Direction() : ra(0.0), dec(0.0) {}

  /**
   * @brief Construct a new Direction object
   *
   * @param ra Right ascension in radians
   * @param dec Declination in radians
   */
  constexpr Direction(double _ra, double _dec) : ra(_ra), dec(_dec) {}

  /**
   * @brief Angular distance
   *
   * @param ra right ascention in radians
   * @param dec declination in radians
   */
  inline double angular_distance(double other_ra, double other_dec) const {
    const double delta_ra = ra - other_ra;
    const double value =
        std::sin(dec) * std::sin(other_dec) +
        std::cos(delta_ra) * std::cos(other_dec) * std::cos(dec);
    // Avoid nan in rounding
    return value <= 1 ? std::acos(value) : 0.0;
  }

  /**
   * @brief Angular distance small angles approximation
   *
   * @param ra right ascention in radians
   * @param dec declination in radians
   */
  inline double angular_distance_approx(double other_ra, double other_dec) {
    const double delta_ra = ra - other_ra;
    const double cos_ra = std::cos(ra);
    const double delta_dec = dec - other_dec;
    return std::sqrt(delta_ra * delta_ra * cos_ra * cos_ra +
                     delta_dec * delta_dec);
  }

  /**
   * @brief Angular distance small angles approximation
   *
   * @param ra right ascention in radians
   * @param dec declination in radians
   */
  inline double angular_distance_square_approx(double other_ra,
                                               double other_dec) {
    const double delta_ra = ra - other_ra;
    const double cos_ra = std::cos(ra);
    const double delta_dec = dec - other_dec;
    return delta_ra * delta_ra * cos_ra * cos_ra + delta_dec * delta_dec;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const Direction &direction) {
    os << "(" << direction.ra << ", " << direction.dec << ")";

    return os;
  }

  double ra;  ///< Right ascension in radians
  double dec; ///< Declination in radians
};

/// @}
} // namespace predict

#endif // PREDICT_DIRECTION_H_
