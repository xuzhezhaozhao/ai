# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ./makefile.inc

LIBS += -L. -lclustering

SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)

all: libclustering.a do_kmeans

libclustering.a: $(OBJ)
	ar r $@ $^

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c -o $@ $<

do_kmeans: do_kmeans.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f libclustering.*
	rm -f $(OBJ)
	rm -rf do_kmeans


.PHONY: all clean run
