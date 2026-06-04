#pragma once

#include <functional>
#include <string>

void print_header(const std::string &title);

void print_timing(const std::string &name, double seconds);

void print_timing(std::pair<const std::string, double> timing);

void print_timing(const std::string &name, double seconds, double percent);

// Time the execution of the lambda given as the second argument, and print it
// to stdout.
// ====================================================
//  C++ version of time_function provided by Jess, 2026
//  <jess.klompmaker@outlook.com>
// ====================================================
int time_function(std::vector<std::pair<const std::string, double>> &timings,
                  std::string message, std::function<void()> function_pointer);
