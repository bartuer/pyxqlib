#ifndef QLIBC_QUOTE_H_
#define QLIBC_QUOTE_H_
#include <algorithm>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace qlibc {
class Quote {
public:
  Quote();
  ~Quote();

  float sum(float *value, size_t beg, size_t end);

private:
  // Disallows copy and assignment.
  Quote(const Quote &);
  Quote &operator=(const Quote &);
};
} // namespace qlibc
#endif // QLIBC_QUOTE_H_