#include "quote.h"
#include <functional>
#include <numeric>

namespace qlibc {
Quote::Quote() {}
Quote::~Quote() {}

float sum(float *value, size_t beg, size_t end) {
  return std::accumulate(std::vector<float>::iterator(value + beg),
                         std::vector<float>::iterator(value + end), 0);
}

} // namespace qlibc