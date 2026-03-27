// ObjectCollection.h: A simple abstact class for a collection of objects
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief A simple abstact class for a collection of objects

#ifndef OBJECT_COLLECTION_H_
#define OBJECT_COLLECTION_H_

#include <cstddef>

template <class T> class ObjectCollection {
public:
  virtual ~ObjectCollection() = default;

  virtual void Add(const T &object) = 0;
  virtual void Clear() = 0;
  virtual size_t Size() const = 0;
  virtual void Reserve(size_t size) = 0;
};

#endif // OBJECT_COLLECTION_H_
