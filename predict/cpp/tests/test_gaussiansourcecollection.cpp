
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#define BOOST_TEST_MODULE GAUSSIAN_SOURCE_COLLECTION_TEST
#include <boost/test/unit_test.hpp>
#include <predict/GaussianSourceCollection.h>

BOOST_AUTO_TEST_SUITE(GaussianSourceCollectionTestSuite)
using namespace predict;
BOOST_AUTO_TEST_CASE(add_single) {
  GaussianSource source(Direction(0.1, 0.2), Stokes(1.0, 0.0, 0.0, 0.0));
  GaussianSourceCollection collection;
  collection.Add(source);
  BOOST_CHECK_EQUAL(collection.Size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.Size(), 1);

  BOOST_CHECK_EQUAL(collection.direction_vector.ra[0], 0.1);
  BOOST_CHECK_EQUAL(collection.direction_vector.dec[0], 0.2);
  BOOST_CHECK_EQUAL(collection.spectra[0].GetReferenceFlux().I, 1.0);
}

BOOST_AUTO_TEST_CASE(reserve) {
  std::vector<PointSource> sources{
      {Direction(0.1, 0.2), Stokes(1.0, 0.0, 0.0, 0.0)},
      {Direction(0.2, 0.3), Stokes(2.0, 0.0, 0.0, 0.0)}};
  PointSourceCollection collection;
  collection.Reserve(2);

  auto ptr = collection.direction_vector.ra.data();

  collection.Add(sources[0]);
  collection.Add(sources[1]);

  BOOST_CHECK_EQUAL(collection.Size(), 2);
  BOOST_CHECK_EQUAL(collection.direction_vector.ra.data(), ptr);
}

BOOST_AUTO_TEST_SUITE_END()