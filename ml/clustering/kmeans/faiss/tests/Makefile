# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

TESTS_SRC = $(wildcard *.cpp)
TESTS     = $(TESTS_SRC:.cpp=.o)

LIBS += -lclustering
LDFLAGS += -L..

all: test_mini

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c -o $@ $<

test_mini: test_mini.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o

.PHONY: all clean
