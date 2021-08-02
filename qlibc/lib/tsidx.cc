#include "tsidx.h"
#include <functional>
#include <numeric>

#define DAY_SECS 86400
namespace qlibc {
TSIdx::TSIdx()
    : d_beg(0), d_end(0), days(0), d_idx(NULL), d_xdi(NULL), t_idx(NULL) {}
TSIdx::~TSIdx() {
  if (d_idx != NULL) {
    free(d_idx);
  }
  if (d_xdi != NULL) {
    free(d_xdi);
  }
  if (t_idx != NULL) {
    free(t_idx);
  }
}

int TSIdx::build(uint32_t *ts, size_t len) {
  uint32_t first = ts[0];
  uint32_t last = ts[len - 1];
  this->d_beg = first / DAY_SECS;
  this->d_end = last / DAY_SECS + 1;
  this->days = this->d_end - this->d_beg;
  this->d_idx = (uint32_t *)malloc(this->days * sizeof(uint32_t));
  assert(this->d_idx);
  if (this->d_idx == NULL) {
    return -1;
  }
  memset(this->d_idx, 0, this->days * sizeof(uint32_t));
  this->d_xdi = (uint32_t *)malloc(this->days * sizeof(uint32_t));
  assert(this->d_xdi);
  if (this->d_xdi == NULL) {
    return -1;
  }
  memset(this->d_xdi, 0, this->days * sizeof(uint32_t));
  this->t_idx = (uint32_t *)malloc(len * sizeof(uint32_t));
  assert(this->t_idx);
  if (this->t_idx == NULL) {
    return -1;
  }
  memcpy(this->t_idx, ts, len);

  for (size_t i = 0, di = 0; i < len; i++) {
    uint32_t t = ts[i];
    uint32_t d = t / DAY_SECS - this->d_beg;
    while (d >= di) {
      assert(di < this->days);
      if (di >= this->days) {
        return -2;
      }
      this->d_idx[di] = i;
      di++;
    }
  }

  for (size_t i = len - 1, di = this->days - 1; i >= 0; i--) {
    uint32_t t = ts[i];
    size_t d = this->d_end - t / DAY_SECS;
    while (di >= d) {
      assert(di >= 0);
      if (di < 0) {
        return -2;
      }
      this->d_xdi[di] = i + 1;
      di--;
    }
  }

  return 0;
}

uint32_t TSIdx::start_search_plain(size_t i, uint32_t s) {
  assert(i + 1 < this->days);
  uint32_t d = this->d_idx[i];
  uint32_t n = this->d_idx[i + 1];
  uint32_t j;
  for (j = d; j < n; j++) {
    if (this->t_idx[j] - s > 0) {
      return j;
    }
  }
  return j;
}

uint32_t TSIdx::start(uint32_t start) {
  uint32_t d = start / DAY_SECS;
  size_t i = d - this->d_beg;
  assert(i < this->days);
  if (start == d * DAY_SECS) {
    return this->d_idx[i];
  } else {
    return this->start_search_plain(i, start);
  }
}

uint32_t TSIdx::stop_search_plain(size_t i, uint32_t s) {
  assert(i - 1 >= 0);
  uint32_t d = this->d_xdi[i];
  uint32_t n = this->d_xdi[i - 1];
  uint32_t j;
  for (j = d; j >= n; j--) {
    if (this->t_idx[j] - s < 0) {
      return j;
    }
  }
  return j;
}

uint32_t TSIdx::stop(uint32_t stop) {
  uint32_t d = stop / DAY_SECS;
  size_t i = d - this->d_beg;
  assert(i == this->d_end - d);
  assert(i < this->days);
  if (stop == d * DAY_SECS) {
    return this->d_xdi[i];
  } else {
    return this->stop_search_plain(i, stop);
  }
}

std::pair<uint32_t, uint32_t> TSIdx::index(uint32_t start, uint32_t stop) {
  return std::make_pair(this->start(start), this->stop(stop));
}
} // namespace qlibc