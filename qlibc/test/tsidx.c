#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif // _WIN32

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <tsidx.h>

#include "cmdopt.h"

namespace {
int param_num = 7;
void print_help(const char *cmd) {
  std::cerr << "Usage: " << cmd
            << " [OPTION]... \n\n"
               "Options:\n"
               "  -n, --num            number\n"
               "  -h, --help           print this help\n"
            << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  std::ios::sync_with_stdio(false);

  ::cmdopt_option long_options[] = {
      {"num", 1, NULL, 'n'}, {"help", 0, NULL, 'h'}, {NULL, 0, NULL, 0}};
  ::cmdopt_t cmdopt;
  ::cmdopt_init(&cmdopt, argc, argv, "n:tbwlc:o:xvgh", long_options);
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
    case 'h': {
      print_help(argv[0]);
      return 0;
    }
    default: { return 1; }
    }
  }
  printf("num :%d\n", param_num);
}