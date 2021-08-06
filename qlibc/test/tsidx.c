#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif // _WIN32

#include "cmdopt.h"
#include "picojson.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <tsidx.h>

inline time_t str2time_t(const char *time_str) {
  struct tm tm;
  strptime(time_str, "%Y-%m-%d %H:%M:%S", &tm);
  return mktime(&tm);
}

inline char *time_t2str(const time_t t, char *time_str) {
  memset(time_str, 0, 20);
  std::strftime(time_str, 20, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return time_str;
}

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

  std::ifstream time_case(case_filename);
  picojson::value c;
  time_case >> c;
  if (time_case.fail()) {
    std::cerr << "JSON parse error -c " << case_filename << ", "
              << picojson::get_last_error() << std::endl;
    return -1;
  }

  std::cout << "ts.size(): " << ts.size() << std::endl;
  std::cout << str2time_t("2020-01-02 09:30:00") << ", " << 1577957400
            << std::endl;

  time_t t = 1577957400;
  char tfstr[20];
  std::cout << time_t2str(t, tfstr) << std::endl;
}