#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <sstream>

#include <boost/test/minimal.hpp>
#include <CL/sycl.hpp>

// We assume we are in build/tests/K-nn
#ifndef KNN_TRAIN_DATA_PATH
#define KNN_TRAIN_DATA_PATH "../../../tests/K-nn/data/trainingsample.csv"
#endif
#ifndef KNN_VALID_DATA_PATH
#define KNN_VALID_DATA_PATH "../../../tests/K-nn/data/validationsample.csv"
#endif

using namespace cl::sycl;

constexpr size_t training_set_size = 5000;
constexpr size_t pixel_number = 784;

using Vector = std::array<int, pixel_number>;

struct Img {
  // The digit value [0-9] represented on the image
  int label;
  // The 1D-linearized image pixels
  Vector pixels;
};

// Construct a SYCL buffer from a vector of images
template<typename T>
buffer<int> get_buffer(const std::vector<T>& imgs) {
  std::vector<int> res;
  for (auto const& elem : imgs) {
    res.insert(res.end(), std::begin(elem.pixels), std::end(elem.pixels));
  }
  return { std::begin(res), std::end(res) };
}

// Read a CSV-file containing image pixels
template<typename T>
std::vector<T> slurp_file(const std::string& name) {
  std::ifstream infile { name, std::ifstream::in };
  std::string line, token;
  std::vector<T> res;
  bool fst_1 = true;

  while (std::getline(infile, line)) {
    if (fst_1) {
      fst_1 = false;
      continue;
    }
    T img;
    std::istringstream iss { line };
    bool fst = true;
    int index = 0;
    while (std::getline(iss, token, ',')) {
      if (fst) {
        img.label = std::stoi(token);
        fst = false;
      }
      else {
        img.pixels[index] = std::stoi(token);
        index++;
      }
    }
    res.push_back(img);
  }
  return res;
}
