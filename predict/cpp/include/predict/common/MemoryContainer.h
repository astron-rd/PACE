// MemoryContainer.h: A memory container that can be mapped to an xtensor
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#ifndef MEMORY_CONTAINER_H_
#define MEMORY_CONTAINER_H_

template <typename U> class ContiguousMemoryIterator {
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = U;
  using pointer = std::add_pointer_t<U>;            // or also value_type*
  using reference = std::add_lvalue_reference_t<U>; // or also value_type&

  ContiguousMemoryIterator(pointer ptr) : ptr_(ptr) {}
  reference operator*() const { return *ptr_; }
  pointer operator->() { return ptr_; }

  // Prefix increment
  ContiguousMemoryIterator &operator++() {
    ptr_++;
    return *this;
  }

  // Postfix increment
  ContiguousMemoryIterator operator++(int) {
    Iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const ContiguousMemoryIterator &a,
                         const ContiguousMemoryIterator &b) {
    return a.ptr_ == b.ptr_;
  };
  friend bool operator!=(const ContiguousMemoryIterator &a,
                         const ContiguousMemoryIterator &b) {
    return a.ptr_ != b.ptr_;
  };

private:
  pointer ptr_;
};

template <typename T> class MemoryContainer {
  struct ConstIterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = const T;
    using pointer = std::add_pointer_t<const T>; // or also value_type*
    using reference =
        std::add_lvalue_reference_t<const T>; // or also value_type&
  };

  using value_type = T;
  using reference = std::add_lvalue_reference_t<T>;
  using const_reference = std::add_pointer_t<const T>;
  using iterator = ContiguousMemoryIterator<T>;
  using const_iterator = ContiguousMemoryIterator<const T>;
  iterator const_iterator using difference_type = T;

  iterator begin() const = 0;
  iterator end() const = 0;
  const_iterator cbegin() const = 0;
  const_iterator cend() const = 0;

  reference operator[](size_t index) const = 0;
  reference front() const = 0;
  reference back() const = 0;

  pointer data() const = 0;

  size_t size() const = 0;
  size_t reshape(size_t new_size) = 0;
  size_t resize(size_t new_size) = 0;
};

#endif // MEMORY_CONTAINER_H_