#include "util.hpp"

// Construct a SYCL buffer from a vector of images
buffer<int> get_buffer(const std::vector<Img>& imgs) {
  std::vector<int> res;
  for (auto const& elem : imgs) {
    res.insert(res.end(), std::begin(elem.pixels), std::end(elem.pixels));
  }
  return { std::begin(res), std::end(res) };
}

// Read a CSV-file containing image pixels
std::vector<Img> slurp_file(const std::string& name) {
  std::ifstream infile { name, std::ifstream::in };
  std::string line, token;
  std::vector<Img> res;
  bool fst_1 = true;

  while (std::getline(infile, line)) {
    if (fst_1) {
      fst_1 = false;
      continue;
    }
    Img img;
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
