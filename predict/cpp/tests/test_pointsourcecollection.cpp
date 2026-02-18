
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#define BOOST_TEST_MODULE POINT_SOURCE_COLLECTION_TEST
#include <boost/test/unit_test.hpp>
#include <predict/PointSourceCollection.h>
using namespace predict;
BOOST_AUTO_TEST_SUITE(PointSourceCollectionTestSuite)

BOOST_AUTO_TEST_CASE(add_single) {
  PointSource source(Direction(0.1, 0.2), Stokes(1.0, 0.0, 0.0, 0.0));
  PointSourceCollection collection;
  collection.Add(source);
  BOOST_CHECK_EQUAL(collection.Size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.Size(), 1);
  BOOST_CHECK_EQUAL(collection.spectra.size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.ra[0], 0.1);
  BOOST_CHECK_EQUAL(collection.direction_vector.dec[0], 0.2);
  BOOST_CHECK_EQUAL(collection.spectra[0].GetReferenceFlux().I, 1.0);
}

BOOST_AUTO_TEST_CASE(add_single_unspecified_beam_id) {
  PointSource source(Direction(0.1, 0.2), Stokes(1.0, 0.0, 0.0, 0.0));
  PointSourceCollection collection;
  collection.Add(source);
  BOOST_CHECK_EQUAL(collection.Size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.Size(), 1);
  BOOST_CHECK_EQUAL(collection.spectra.size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.ra[0], 0.1);
  BOOST_CHECK_EQUAL(collection.direction_vector.dec[0], 0.2);

  BOOST_CHECK_EQUAL(collection.beam_id[0], 0);
}

BOOST_AUTO_TEST_CASE(add_single_with_beam_id) {
  PointSource source(Direction(0.1, 0.2), Stokes(1.0, 0.0, 0.0, 0.0), 15);
  PointSourceCollection collection;
  collection.Add(source);
  BOOST_CHECK_EQUAL(collection.Size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.Size(), 1);
  BOOST_CHECK_EQUAL(collection.spectra.size(), 1);
  BOOST_CHECK_EQUAL(collection.direction_vector.ra[0], 0.1);
  BOOST_CHECK_EQUAL(collection.direction_vector.dec[0], 0.2);
  BOOST_CHECK_EQUAL(collection.spectra[0].GetReferenceFlux().I, 1.0);
  BOOST_CHECK_EQUAL(collection.beam_id[0], 15);
  BOOST_CHECK_EQUAL(collection.unique_beam_ids.size(), 1);
}

BOOST_AUTO_TEST_CASE(add_multiple_with_beam_id) {
  std::vector<PointSource> sources{
      {Direction(0.1, 0.2), Stokes(1.0, 0.0, 0.0, 0.0), 15},
      {Direction(0.2, 0.3), Stokes(2.0, 0.0, 0.0, 0.0), 15},
      {Direction(0.3, 0.4), Stokes(3.0, 0.0, 0.0, 0.0), 16}};
  PointSourceCollection collection;
  collection.Add(sources[0]);
  collection.Add(sources[1]);
  collection.Add(sources[2]);

  BOOST_CHECK_EQUAL(collection.unique_beam_ids.size(), 2);
  BOOST_CHECK(collection.unique_beam_ids.contains(16));
  BOOST_CHECK(collection.unique_beam_ids.contains(15));
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