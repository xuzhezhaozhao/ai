
CXX          = g++ -std=c++11
CPPFLAGS     = -DFINTEGER=int
CXXFLAGS     = -fPIC -fopenmp -m64 -Wno-sign-compare -g -O3 -Wall -Wextra
CPUFLAGS     = -msse4 -mpopcnt
LDFLAGS      = -fopenmp
LIBS         = -lblas -llapack
