#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif // _WIN32
#include "cmdopt.h"
#include "picojson.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <tsidx.h>

using namespace std::chrono;

const char *time_stamp_format = "%Y-%m-%d %H:%M:%S";

inline time_t str2time_t(const char *time_str) {
  struct tm tm;
  strptime(time_str, time_stamp_format, &tm);
  return mktime(&tm);
}

inline char *time_t2str(const time_t t, char *time_str) {
  memset(time_str, 0, 20);
  std::strftime(time_str, 20, time_stamp_format, std::localtime(&t));
  return time_str;
}

inline std::string time_t2strw(const time_t t) {
  std::string str = std::asctime(std::localtime(&t));
  str.pop_back();
  return str;
}

class Clock {
public:
  Clock() : beg_(high_resolution_clock::now()) {}

  inline void reset() { beg_ = high_resolution_clock::now(); }

  inline double elasped() const {
    duration<double, std::nano> duration = high_resolution_clock::now() - beg_;
    return duration.count();
  }

private:
  time_point<system_clock> beg_;
};

inline void print_start(double elaspe, uint32_t start, time_t &input,
                        std::vector<uint32_t> &time_series, char *output) {

  return;
  std::cout << start << ", " << time_t2strw(input) << " -> "
            << time_t2str(time_series[start], output) << " [" << elaspe
            << "ns]: " << std::endl;
}

inline void print_stop(double elaspe, uint32_t stop, time_t &input,
                       std::vector<uint32_t> &time_series, char *output) {
  return;
  std::cout << stop << ", " << time_t2str(time_series[stop], output) << " <- "
            << time_t2strw(input) << " [" << elaspe << "ns]: " << std::endl;
}

enum test_t { start_test = 0, stop_test = 1 };
union timestamp_t {
  time_t start;
  time_t stop;
};

typedef struct {
  enum test_t type;
  char name[32];
  union timestamp_t timestamp;
  uint32_t expect;
  bool result;
  double duration;
} test_case;

namespace {
int param_num = 7;
const char *data_filename = NULL;
const char *case_filename = NULL;
void print_help(const char *cmd) {
  std::cerr << "Usage: " << cmd
            << " [OPTION]... \n\n"
               "Options:\n"
               "  -n, --num            number\n"
               "  -h, --help           print this help\n"
               "  -d, --data           time series data file name\n"
               "  -c, --test_case      test case file name\n"
            << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  std::ios::sync_with_stdio(false);

  ::cmdopt_option long_options[] = {{"num", 1, NULL, 'n'},
                                    {"data", 1, NULL, 'd'},
                                    {"case", 1, NULL, 'd'},
                                    {"help", 0, NULL, 'h'},
                                    {NULL, 0, NULL, 0}};
  ::cmdopt_t cmdopt;
  ::cmdopt_init(&cmdopt, argc, argv, "n:d:c:h", long_options);
  int label;
  while ((label = ::cmdopt_get(&cmdopt)) != -1) {
    switch (label) {
    case 'n': {
      char *end_of_value;
      const long value = std::strtol(cmdopt.optarg, &end_of_value, 10);
      if ((*end_of_value != 0) || (value <= 0) || (value > 20)) {
        std::cerr << "error: option `-n' with an invalid argument: "
                  << cmdopt.optarg << std::endl;
        return 1;
      }
      param_num = (int)value;
      break;
    }
    case 'd': {
      data_filename = cmdopt.optarg;
      break;
    }
    case 'c': {
      case_filename = cmdopt.optarg;
      break;
    }
    case 'h': {
      print_help(argv[0]);
      return 0;
    }
    default: { return 1; }
    }
  }

  std::ifstream time_series(data_filename);
  picojson::value v;
  time_series >> v;
  if (time_series.fail()) {
    std::cerr << "JSON parse error -d " << data_filename << ", "
              << picojson::get_last_error() << std::endl;
    return -1;
  }

  std::vector<uint32_t> ts;
  if (v.is<picojson::array>()) {
    const picojson::array &a = v.get<picojson::array>();
    size_t sz = a.size();
    ts.reserve(sz);
    for (picojson::array::const_iterator i = a.begin(); i != a.end(); ++i) {
      ts.push_back(i->u_.number_);
    }
  }

  qlibc::TSIdx *tsidx = new qlibc::TSIdx();
  tsidx->build(ts.data(), ts.size());
  char tfstr[20];

  std::ifstream time_case(case_filename);
  picojson::value tests;
  time_case >> tests;
  if (time_case.fail()) {
    std::cerr << "JSON parse error -c " << case_filename << ", "
              << picojson::get_last_error() << std::endl;
    return -1;
  }

  std::vector<test_case> tcs;
  if (tests.is<picojson::array>()) {
    const picojson::array &cs = tests.get<picojson::array>();
    for (picojson::array::const_iterator j = cs.begin(); j != cs.end(); ++j) {
      const picojson::value::object &c = j->get<picojson::object>();
      test_case tc;
      for (picojson::value::object::const_iterator i = c.begin(); i != c.end();
           i++) {
        if (i->first == "start") {
          tc.timestamp.start = str2time_t(i->second.to_str().c_str());
        } else if (i->first == "stop") {
          tc.timestamp.stop = str2time_t(i->second.to_str().c_str());
        } else if (i->first == "type") {
          if (i->second.to_str() == "start") {
            tc.type = start_test;
          } else if (i->second.to_str() == "stop") {
            tc.type = stop_test;
          } else {
            std::cerr << "test case JSON parse error, type should be one of "
                         "start and stop"
                      << std::endl;
            return -2;
          }
        } else if (i->first == "expect") {
          tc.expect = i->second.get<double>();
        } else if (i->first == "name") {
          memset(tc.name, 0, 32);
          strncpy(tc.name, i->second.to_str().c_str(), 31);
        } else {
          std::cerr << "test case JSON parse error, field should be one of "
                       "start, stop, name, expect and type"
                    << std::endl;
          return -2;
        }
      }
      tc.result = false;
      tc.duration = 0.0;
      tcs.push_back(tc);
    }
  }

  time_t tt = str2time_t("2020-05-31 07:30:00");
  Clock cl;
  uint32_t start = tsidx->start(tt);
  double e = cl.elasped();
  print_start(e, start, tt, ts, tfstr);

  cl.reset();
  uint32_t stop = tsidx->stop(tt);
  e = cl.elasped();
  print_stop(e, stop, tt, ts, tfstr);

  tt = str2time_t("2020-06-01 12:30:00");
  cl.reset();
  start = tsidx->start(tt);
  e = cl.elasped();
  print_start(e, start, tt, ts, tfstr);

  picojson::array &cs = tests.get<picojson::array>();
  std::vector<test_case>::const_iterator i;
  picojson::array::iterator j;
  for (i = tcs.begin(), j = cs.begin(); j != cs.end(); ++j, ++i) {
    picojson::value::object &c = j->get<picojson::object>();
    test_case tc = *i;
    c["duration"] = picojson::value(tc.duration);
    c["result"] = picojson::value(tc.result);
  }

  std::string json_str = tests.serialize();
  std::cout << json_str << std::endl;
}