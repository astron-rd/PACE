// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0
#include <span>

#include <predict/Directions.h>
#include <predict/Predict.h>
#include <xsimd/xsimd.hpp>
#include <xtensor/xadapt.hpp>

#define BOOST_TEST_MODULE DIRECTIONS_TEST
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>
using namespace predict;

BOOST_AUTO_TEST_SUITE(DirectionsTestSuite)

BOOST_AUTO_TEST_CASE(test_add_single_direction) {
  Directions directions;
  directions.Add(Direction(0.1, 0.2));
  BOOST_CHECK_EQUAL(directions.Size(), 1);
  BOOST_CHECK_EQUAL(directions.ra[0], 0.1);
  BOOST_CHECK_EQUAL(directions.dec[0], 0.2);
}

BOOST_AUTO_TEST_CASE(test_add_multi_direction) {
  Directions directions;
  std::vector<Direction> dir_vec = {Direction(0.1, 0.2), Direction(0.3, 0.4)};
  directions.Add(std::span<Direction>(dir_vec.data(), dir_vec.size()));
  BOOST_CHECK_EQUAL(directions.Size(), 2);
  BOOST_CHECK_EQUAL(directions.ra[0], 0.1);
  BOOST_CHECK_EQUAL(directions.dec[0], 0.2);

  BOOST_CHECK_EQUAL(directions.ra[1], 0.3);
  BOOST_CHECK_EQUAL(directions.dec[1], 0.4);
}

BOOST_AUTO_TEST_CASE(test_reserve) {
  Directions directions;
  std::vector<Direction> dir_vec = {Direction(0.1, 0.2), Direction(0.3, 0.4)};
  directions.Reserve(2);
  BOOST_CHECK_EQUAL(directions.Size(), 0);
  auto ptr = directions.ra.data();
  directions.Add(std::span<Direction>(dir_vec.data(), dir_vec.size()));
  BOOST_CHECK_EQUAL(directions.Size(), 2);
  BOOST_CHECK_EQUAL(directions.ra.data(), ptr);
  std::vector<double> ra{0.60, 0.70};

  auto bla = xt::adapt(ra.data(), ra.size(), xt::no_ownership());
}

BOOST_AUTO_TEST_CASE(test_radec_to_lmn_compare_reference) {
  Directions dirs;

  Direction reference(0.2, 0.3);
  Direction dir_0(0.1, 0.2);
  Direction dir_1(0.21, 0.3);

  dirs.Add(dir_0);
  dirs.Add(dir_1);

  xt::xtensor<double, 2> lmn_expected({2, 3});

  radec2lmn(reference, dir_0, &lmn_expected(0, 0));
  radec2lmn(reference, dir_1, &lmn_expected(1, 0));

  xt::xtensor<double, 2> lmn_actual({2, 3});
  dirs.RaDec2Lmn<Directions::SINGLE>(reference, lmn_actual);
  for (size_t i = 0; i < 2; ++i) {
    BOOST_CHECK_CLOSE(lmn_expected(i, 0), lmn_actual(i, 0), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 1), lmn_actual(i, 1), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 2), lmn_actual(i, 2), 1.e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_radec_to_lmn_multi) {
  Directions dirs;
  Direction reference(0.2, 0.3);
  Direction dir_0(0.1, 0.2);
  Direction dir_1(0.21, 0.3);

  dirs.Add(dir_0);
  dirs.Add(dir_1);

  xt::xtensor<double, 2> lmn_expected({2, 3});
  xt::xtensor<double, 2> lmn_actual({2, 3});

  dirs.RaDec2Lmn<Directions::SINGLE>(reference, lmn_expected);
  dirs.RaDec2Lmn<Directions::MULTI>(reference, lmn_actual);
  for (size_t i = 0; i < 2; ++i) {
    BOOST_CHECK_CLOSE(lmn_expected(i, 0), lmn_actual(i, 0), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 1), lmn_actual(i, 1), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 2), lmn_actual(i, 2), 1.e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_radec_to_lmn_xsimd) {
  Directions dirs;

  Direction reference(0.2, 0.3);
  Direction dir_0(0.1, 0.2);
  Direction dir_1(0.21, 0.3);

  dirs.Add(dir_0);
  dirs.Add(dir_1);

  xt::xtensor<double, 2> lmn_expected({2, 3});
  xt::xtensor<double, 2> lmn_actual({2, 3});

  dirs.RaDec2Lmn<Directions::computation_strategy::SINGLE>(reference,
                                                           lmn_expected);

  dirs.RaDec2Lmn<Directions::computation_strategy::XSIMD>(reference,
                                                          lmn_actual);
  for (size_t i = 0; i < 2; ++i) {
    BOOST_CHECK_CLOSE(lmn_expected(i, 0), lmn_actual(i, 0), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 1), lmn_actual(i, 1), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 2), lmn_actual(i, 2), 1.e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_radec_to_lmn_xsimd_with_remainder) {
  Direction reference(0.2, 0.3);
  Direction dir_0(0.1, 0.2);
  Direction dir_1(0.21, 0.3);
  Direction dir_2(0.3, 0.4);

  xt::xtensor<double, 2> lmn_expected({3, 3});

  radec2lmn(reference, dir_0, &lmn_expected(0, 0));
  radec2lmn(reference, dir_1, &lmn_expected(1, 0));
  radec2lmn(reference, dir_2, &lmn_expected(2, 0));

  Directions directions;
  directions.Add(dir_0);
  directions.Add(dir_1);
  directions.Add(dir_2);

  xt::xtensor<double, 2> lmn_actual({3, 3});
  directions.RaDec2Lmn<Directions::computation_strategy::XSIMD>(reference,
                                                                lmn_actual);
  for (size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(lmn_expected(i, 0), lmn_actual(i, 0), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 1), lmn_actual(i, 1), 1.e-6);
    BOOST_CHECK_CLOSE(lmn_expected(i, 2), lmn_actual(i, 2), 1.e-6);
  }
}

BOOST_AUTO_TEST_SUITE_END()