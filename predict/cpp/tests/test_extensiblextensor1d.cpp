
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#define BOOST_TEST_MODULE EXTENSIBLE_XTENSOR_TEST
#include <boost/test/unit_test.hpp>
#include <predict/common/ExtensibleXtensor1D.h>

BOOST_AUTO_TEST_SUITE(ExtensibleXtensorTestSuite)

BOOST_AUTO_TEST_CASE(test_add_element) {
  ExtensibleXtensor1D<int> tensor{1, 2, 4};

  BOOST_CHECK(tensor.size() == 3);

  tensor.push_back(5);
  BOOST_CHECK(tensor.size() == 4);
  BOOST_CHECK(tensor(3) == 5);
}

BOOST_AUTO_TEST_CASE(test_resize) {
  ExtensibleXtensor1D<int> tensor{1, 2, 4};

  tensor.resize(5);
  BOOST_CHECK(tensor.max_size() == 5);
  BOOST_CHECK(tensor.size() == 5);
}

BOOST_AUTO_TEST_CASE(test_avoid_resize) {
  ExtensibleXtensor1D<int> tensor{1, 2, 4};

  tensor.reserve(5);
  BOOST_CHECK(tensor.max_size() == 3);
  BOOST_CHECK(tensor.size() == 5);
  BOOST_CHECK_EQUAL(tensor(0), 1);
  BOOST_CHECK_EQUAL(tensor(1), 2);
  BOOST_CHECK_EQUAL(tensor(2), 4);
  tensor.expand(4);
  BOOST_CHECK(tensor.size() == 5);
}

BOOST_AUTO_TEST_CASE(test_clear) {
  ExtensibleXtensor1D<int> tensor{1, 2, 4};

  tensor.clear();
  BOOST_CHECK(tensor.size() == 0);
}

BOOST_AUTO_TEST_SUITE_END()