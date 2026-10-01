#include "util.h"
#include <chrono>
#include <iomanip>
#include <iostream>

void print_header(const std::string &title) {
  std::cout << "\n==================================================\n";
  if (!title.empty()) {
    std::cout << title << "\n";
    std::cout << "==================================================\n";
  }
}

void print_timing(const std::string &name, double seconds) {
  std::cout << std::left << std::setw(38) << name << std::right << std::setw(10)
            << std::fixed << std::setprecision(6) << seconds << " s\n";
}

void print_timing(std::pair<const std::string, double> timing) {
  print_timing(timing.first, timing.second);
}

void print_timing(const std::string &name, double seconds, double percent) {
  std::cout << std::left << std::setw(31) << name << std::fixed
            << std::setprecision(3) << std::right << std::setw(8) << seconds
            << " s (" << std::setw(5) << std::setprecision(1) << percent
            << "%)\n";
}

// Time the execution of the lambda given as the second argument, and print it
// to stdout.
// ====================================================
//  C++ version of time_function provided by Jess, 2026
//  <jess.klompmaker@outlook.com>
// ====================================================
int time_function(std::vector<std::pair<const std::string, double>> &timings,
                  std::string message, std::function<void()> function_pointer) {
  auto t_start = std::chrono::high_resolution_clock::now();
  function_pointer();
  auto t_end = std::chrono::high_resolution_clock::now();
  float duration = std::chrono::duration<float>(t_end - t_start).count();
  timings.push_back({message, duration});

  print_timing(message, duration);
  return duration;
}
