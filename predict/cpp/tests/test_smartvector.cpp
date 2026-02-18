
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#define BOOST_TEST_MODULE SMART_VECTOR_TEST
#include <boost/test/unit_test.hpp>
#include <predict/common/SmartVector.h>

BOOST_AUTO_TEST_SUITE(SmartVectorTestSuite)

BOOST_AUTO_TEST_CASE(test_add_element) {
  SmartVector<int> tensor{1, 2, 4};

  BOOST_CHECK(tensor.size() == 3);

  tensor.push_back(5);
  BOOST_CHECK_EQUAL(tensor.size(), 4);
  BOOST_CHECK_EQUAL(tensor.view()(3), 5);
  BOOST_CHECK_EQUAL(tensor[3], 5);
}

BOOST_AUTO_TEST_CASE(test_resize_element_keep_content) {
  SmartVector<int> tensor{1, 2, 4};

  BOOST_CHECK_EQUAL(tensor.size(), 3);

  tensor.resize(5);
  BOOST_CHECK_EQUAL(tensor.size(), 5);

  BOOST_CHECK_EQUAL(tensor[2], 4);
}

BOOST_AUTO_TEST_CASE(test_xtensor_access) {
  SmartVector<int> tensor{1, 2, 4};

  BOOST_CHECK_EQUAL(tensor.size(), 3);

  tensor.resize(5);
  SmartVectorView<int> view = tensor.view();

  BOOST_CHECK_EQUAL(view(2), 4);
  view(2) = 10;
  BOOST_CHECK_EQUAL(view(2), 10);
  BOOST_CHECK_EQUAL(tensor[2], 10);

  SmartVectorView<int> casted_view = static_cast<SmartVectorView<int>>(tensor);
  BOOST_CHECK_EQUAL(casted_view.data(), tensor.data());

  casted_view(2) = 4;
  BOOST_CHECK_EQUAL(view(2), 4);
  casted_view(2) = 10;
  BOOST_CHECK_EQUAL(casted_view(2), 10);
  BOOST_CHECK_EQUAL(view(2), 10);
  BOOST_CHECK_EQUAL(tensor[2], 10);
}

BOOST_AUTO_TEST_CASE(test_xtensor_copy) {
  SmartVector<int> tensor{1, 2, 4};

  tensor.push_back(5);

  std::unique_ptr<xt::xtensor<int, 1>> tensor_copy =
      static_cast<std::unique_ptr<xt::xtensor<int, 1>>>(tensor);

  BOOST_CHECK_NE(tensor.data(), tensor_copy->data());

  BOOST_CHECK_EQUAL(tensor_copy->size(), 4);
  BOOST_CHECK_EQUAL(tensor_copy->at(3), 5);

  tensor_copy->at(3) = 10;
  BOOST_CHECK_EQUAL(tensor_copy->at(3), 10);
  BOOST_CHECK_EQUAL(tensor[3], 5); // original tensor should not be modified
}

BOOST_AUTO_TEST_SUITE_END()
