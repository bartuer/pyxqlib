#include "tsidx.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

#define DAY_SECS 86400
namespace qlibc {
TSIdx::TSIdx()
    : d_beg(0), d_end(0), days(0), size(0), d_idx(NULL), d_xdi(NULL),
      t_idx(NULL) {}

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
  this->size = len;
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
  this->t_idx = (int *)malloc(len * sizeof(int));
  assert(this->t_idx);
  if (this->t_idx == NULL) {
    return -1;
  }
  memcpy(this->t_idx, ts, len * sizeof(int));
  for (int i = 0, di = 0; i < this->size; i++) {
    uint32_t t = ts[i];
    int d = t / DAY_SECS - this->d_beg;
    uint32_t dt = (this->d_beg + d) * DAY_SECS;
    if (d_index.size() == 0 || dt != d_index.back()) {
      this->d_index.push_back(dt);
    }
    while (d >= di) {
      assert(di < this->days);
      if (di >= this->days) {
        return -2;
      }
      this->d_idx[di] = i;
      di++;
    }
  }

  for (size_t i = 0; i < this->d_index.size(); i++) {
    this->d_range.push_back(this->start(this->d_index[i]));
  }

  int di = this->days - 1;
  for (int i = len - 1; i >= 0; i--) {
    uint32_t t = ts[i];
    int d = t / DAY_SECS - this->d_beg;
    while (di >= d) {
      if (di < 0) {
        return -2;
      }
      this->d_xdi[di] = i + 1;
      if (di == 0) {
        return 0;
      }
      di--;
    }
  }

  return 0;
}

uint32_t TSIdx::start_search_plain(size_t i, int s) {
  uint32_t d = this->d_idx[i];
  uint32_t n;
  if (this->days - i == 1) {
    n = static_cast<uint32_t>(this->size);
  } else {
    n = this->d_idx[i + 1];
  }
  uint32_t j;
  for (j = d; j < n; j++) {
    if (this->t_idx[j] - s >= 0) {
      return j;
    }
  }
  return j;
}

uint32_t TSIdx::start(uint32_t start) {
  int d = start / DAY_SECS;
  if (d < this->d_beg) {
    return 0;
  }
  int i = std::max(0, d - this->d_beg);
  if (i >= this->days) {
    return this->size;
  }
  i = std::min(i, this->days - 1);
  if (start - d * DAY_SECS == 0) {
    return this->d_idx[i];
  } else {
    return this->start_search_plain(i, start);
  }
}

uint32_t TSIdx::dstart(uint32_t start) {
  int d = start / DAY_SECS;
  size_t dlen = this->d_index.size();
  if (d < this->d_beg) {
    return 0;
  }
  int i = std::max(0, d - this->d_beg);
  if (i >= this->days) {
    return dlen;
  }
  uint32_t j;
  for (j = 0; j < dlen; j++) {
    if (this->d_index[j] - start >= 0) {
      return j;
    }
  }
  return j;
}

uint32_t TSIdx::stop_search_plain(size_t i, int s) {
  uint32_t d = this->d_xdi[i];
  uint32_t n;
  if (i == 0) {
    n = 0;
  } else {
    n = this->d_xdi[i - 1];
  }
  uint32_t j;
  if (d == n) {
    return d;
  }
  for (j = d - 1; j >= n; j--) {
    if (this->t_idx[j] - s <= 0) {
      return j + 1;
    }
  }
  return j + 1;
}

uint32_t TSIdx::stop(uint32_t stop) {
  int d = stop / DAY_SECS;
  if (d > this->d_end) {
    return this->size;
  }
  int i = std::min(d - this->d_beg, this->days - 1);
  if (i < 0) {
    return 0;
  }
  if (stop - d * DAY_SECS == 0) {
    return this->d_xdi[i];
  } else {
    return this->stop_search_plain(i, stop);
  }
}

uint32_t TSIdx::dstop(uint32_t stop) {
  int d = stop / DAY_SECS;
  size_t dlen = this->d_index.size();
  if (d > this->d_end) {
    return dlen;
  }
  int i = std::min(d - this->d_beg, this->days - 1);
  if (i < 0) {
    return 0;
  }
  uint32_t j;
  for (j = dlen - 1; j >= 0; j--) {
    if (this->d_index[j] - stop <= 0) {
      return j + 1;
    }
  }
  return j + 1;
}

std::pair<uint32_t, uint32_t> TSIdx::index(uint32_t start, uint32_t stop) {
  return std::make_pair(this->start(start), this->stop(stop));
}
} // namespace qlibc