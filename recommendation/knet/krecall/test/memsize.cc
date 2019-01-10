#include <sys/sysinfo.h>
#include <stdio.h>

long long AvailableRam() {
  struct sysinfo info;
  int err = sysinfo(&info);
  if (err == 0) {
    return info.freeram;
    return -1;
  }
}

int main(int argc, char *argv[]) {
  long long memsize = AvailableRam();
  printf("memsize = %lld\n", memsize);
  return 0;
}
