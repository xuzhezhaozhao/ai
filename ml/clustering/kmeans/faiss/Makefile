# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ./makefile.inc

SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)

all: libclustering.a

libclustering.a: $(OBJ)
	ar r $@ $^

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c -o $@ $<

clean:
	rm -f libclustering.*
	rm -f $(OBJ)


.PHONY: all clean run
