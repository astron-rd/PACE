
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#define BOOST_TEST_MODULE SPECTRUM_TEST
#include <boost/test/unit_test.hpp>
#include <predict/Direction.h>
#include <predict/PointSource.h>
#include <predict/Spectrum.h>

BOOST_AUTO_TEST_SUITE(SpectrumTestSuite)
using namespace predict;
struct SpectrumFixture {
  SpectrumFixture() {
    spectrum.SetSpectralTerms(1.0e6, false,
                              xt::xtensor<double, 1>({1.0, 2.0, 3.0}));
    spectrum.SetReferenceFlux({1.0, 0.0, 0.0, 0.0});
    spectrum.SetPolarization(0.3, 0.2);
    spectrum.SetRotationMeasure(0.1, false);
  }
  ~SpectrumFixture() { BOOST_TEST_MESSAGE("teardown fixture"); }

  Spectrum spectrum;
  xt::xtensor<double, 1> frequencies = {15.0e6, 20.0e6, 30.0e6};
};

BOOST_FIXTURE_TEST_CASE(normal_no_rotation, SpectrumFixture) {
  PointSource point_source(Direction(0.0, 0.0), spectrum);

  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());

  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  StokesVector result;
  spectrum.Evaluate(frequencies, result);

  BOOST_CHECK_CLOSE(result.I[0], stokes_nu0.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.I[1], stokes_nu1.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.I[2], stokes_nu2.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[0], stokes_nu0.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[1], stokes_nu1.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[2], stokes_nu2.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[0], stokes_nu0.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[1], stokes_nu1.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[2], stokes_nu2.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[0], stokes_nu0.V, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[1], stokes_nu1.V, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[2], stokes_nu2.V, 1.e-6);
}

BOOST_FIXTURE_TEST_CASE(normal_no_rotation_no_spectral_terms, SpectrumFixture) {
  spectrum.ClearSpectralTerms();
  PointSource point_source(Direction(0.0, 0.0), spectrum.GetReferenceFlux());
  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());
  point_source.SetRotationMeasure(0.0, 0.0, 0.0);

  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  StokesVector result;
  spectrum.Evaluate(frequencies, result);

  BOOST_CHECK_CLOSE(result.I[0], stokes_nu0.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.I[1], stokes_nu1.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.I[2], stokes_nu2.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[0], stokes_nu0.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[1], stokes_nu1.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[2], stokes_nu2.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[0], stokes_nu0.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[1], stokes_nu1.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[2], stokes_nu2.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[0], stokes_nu0.V, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[1], stokes_nu1.V, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[2], stokes_nu2.V, 1.e-6);
}

BOOST_FIXTURE_TEST_CASE(normal_with_rotation, SpectrumFixture) {
  spectrum.SetRotationMeasure(spectrum.GetRotationMeasure(), true);

  PointSource point_source(Direction(0.0, 0.0), spectrum.GetReferenceFlux());

  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());
  point_source.SetRotationMeasure(spectrum.GetPolarizationFactor(),
                                  spectrum.GetPolarizationAngle(),
                                  spectrum.GetRotationMeasure());
  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  StokesVector result;
  spectrum.Evaluate(frequencies, result);

  BOOST_CHECK_CLOSE(result.I[0], stokes_nu0.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.I[1], stokes_nu1.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.I[2], stokes_nu2.I, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[0], stokes_nu0.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[1], stokes_nu1.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.Q[2], stokes_nu2.Q, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[0], stokes_nu0.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[1], stokes_nu1.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.U[2], stokes_nu2.U, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[0], stokes_nu0.V, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[1], stokes_nu1.V, 1.e-6);
  BOOST_CHECK_CLOSE(result.V[2], stokes_nu2.V, 1.e-6);
}

BOOST_FIXTURE_TEST_CASE(normal_log_no_rotation, SpectrumFixture) {
  spectrum.SetRotationMeasure(spectrum.GetRotationMeasure(), false);
  spectrum.SetSpectralTerms(spectrum.GetReferenceFrequency(), true,
                            xt::xtensor<double, 1>({1.0e-6, 2.0e-6, 3.0e-6}));

  PointSource point_source(Direction(0.0, 0.0), spectrum.GetReferenceFlux());

  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());
  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  StokesVector result;
  spectrum.Evaluate(frequencies, result);

  BOOST_CHECK_CLOSE_FRACTION(result.I[0], stokes_nu0.I, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.I[1], stokes_nu1.I, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.I[2], stokes_nu2.I, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.Q[0], stokes_nu0.Q, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.Q[1], stokes_nu1.Q, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.Q[2], stokes_nu2.Q, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.U[0], stokes_nu0.U, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.U[1], stokes_nu1.U, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.U[2], stokes_nu2.U, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.V[0], stokes_nu0.V, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.V[1], stokes_nu1.V, 1.e-3);
  BOOST_CHECK_CLOSE_FRACTION(result.V[2], stokes_nu2.V, 1.e-3);
}

BOOST_FIXTURE_TEST_CASE(normal_log_with_rotation, SpectrumFixture) {
  spectrum.SetSpectralTerms(spectrum.GetReferenceFrequency(), true,
                            xt::xtensor<double, 1>({1.0e-6, 2.0e-6, 3.0e-6}));
  spectrum.SetRotationMeasure(spectrum.GetRotationMeasure(), true);

  PointSource point_source(Direction(0.0, 0.0), spectrum.GetReferenceFlux());

  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());

  point_source.SetRotationMeasure(spectrum.GetPolarizationFactor(),
                                  spectrum.GetPolarizationAngle(),
                                  spectrum.GetRotationMeasure());
  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  StokesVector result;
  spectrum.Evaluate(frequencies, result);

  BOOST_CHECK_CLOSE_FRACTION(result.I[0], stokes_nu0.I, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.I[1], stokes_nu1.I, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.I[2], stokes_nu2.I, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.Q[0], stokes_nu0.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.Q[1], stokes_nu1.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.Q[2], stokes_nu2.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.U[0], stokes_nu0.U, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.U[1], stokes_nu1.U, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.U[2], stokes_nu2.U, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.V[0], stokes_nu0.V, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.V[1], stokes_nu1.V, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result.V[2], stokes_nu2.V, 1.e-5);
}

BOOST_FIXTURE_TEST_CASE(normal_log_with_rotation_cross, SpectrumFixture) {
  spectrum.SetSpectralTerms(spectrum.GetReferenceFrequency(), true,
                            xt::xtensor<double, 1>({1.0e-6, 2.0e-6, 3.0e-6}));
  spectrum.SetRotationMeasure(spectrum.GetRotationMeasure(), true);

  PointSource point_source(Direction(0.0, 0.0), spectrum.GetReferenceFlux());

  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());

  point_source.SetRotationMeasure(spectrum.GetPolarizationFactor(),
                                  spectrum.GetPolarizationAngle(),
                                  spectrum.GetRotationMeasure());
  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  xt::xtensor<double, 3> result_complex;
  spectrum.EvaluateCrossCorrelations(frequencies, result_complex);

  BOOST_CHECK_CLOSE_FRACTION(result_complex(0, 0, 0),
                             stokes_nu0.I + stokes_nu0.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result_complex(0, 0, 1),
                             stokes_nu1.I + stokes_nu1.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result_complex(0, 0, 2),
                             stokes_nu2.I + stokes_nu2.Q, 1.e-5);
}

BOOST_FIXTURE_TEST_CASE(normal_with_rotation_cross, SpectrumFixture) {
  spectrum.SetSpectralTerms(spectrum.GetReferenceFrequency(), false,
                            xt::xtensor<double, 1>({1.0e-6, 2.0e-6, 3.0e-6}));
  spectrum.SetRotationMeasure(spectrum.GetRotationMeasure(), true);

  PointSource point_source(Direction(0.0, 0.0), spectrum.GetReferenceFlux());

  point_source.SetSpectralTerms(
      spectrum.GetReferenceFrequency(), spectrum.HasLogarithmicSpectralIndex(),
      spectrum.GetSpectralTerms().begin(), spectrum.GetSpectralTerms().end());

  point_source.SetRotationMeasure(spectrum.GetPolarizationFactor(),
                                  spectrum.GetPolarizationAngle(),
                                  spectrum.GetRotationMeasure());
  Stokes stokes_nu0(0, 0, 0, 0);
  Stokes stokes_nu1(0, 0, 0, 0);
  Stokes stokes_nu2(0, 0, 0, 0);

  stokes_nu0 = point_source.GetStokes(frequencies[0]);
  stokes_nu1 = point_source.GetStokes(frequencies[1]);
  stokes_nu2 = point_source.GetStokes(frequencies[2]);

  xt::xtensor<double, 3> result_complex;
  spectrum.EvaluateCrossCorrelations(frequencies, result_complex);

  BOOST_CHECK_CLOSE_FRACTION(result_complex(0, 0, 0),
                             stokes_nu0.I + stokes_nu0.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result_complex(0, 0, 1),
                             stokes_nu1.I + stokes_nu1.Q, 1.e-5);
  BOOST_CHECK_CLOSE_FRACTION(result_complex(0, 0, 2),
                             stokes_nu2.I + stokes_nu2.Q, 1.e-5);
}

BOOST_AUTO_TEST_SUITE_END()