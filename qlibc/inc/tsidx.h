#ifndef QLIBC_TSIDX_H_
#define QLIBC_TSIDX_H_
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace qlibc {
class TSIdx {
public:
  TSIdx();
  ~TSIdx();

  /* success return 0 */
  int build(uint32_t *ts, size_t len);
  /* start == 0 means only take  */
  std::pair<uint32_t, uint32_t> index(uint32_t start, uint32_t stop);
  uint32_t stop(uint32_t stop);
  uint32_t start(uint32_t start);
  inline uint32_t start_search_plain(size_t i, uint32_t s);
  inline uint32_t stop_search_plain(size_t i, uint32_t s);
  /* inline uint32_t start_search_bin(size_t i, uint32_t s); */
  /* inline uint32_t stop_search_bin(size_t i, uint32_t s); */
  /* inline uint32_t start_search_sse(size_t i, uint32_t s); */
  /* inline uint32_t stop_search_sse(size_t i, uint32_t s); */
private:
  size_t d_beg;
  size_t d_end;
  size_t days;
  uint32_t *d_idx;
  uint32_t *d_xdi;
  uint32_t *t_idx;
};
} // namespace qlibc
#endif // QLIBC_TSIDX_H_