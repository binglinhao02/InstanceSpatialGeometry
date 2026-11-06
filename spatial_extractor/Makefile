# Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++11 -fPIC -Wall -Wextra -Wpedantic

all: libgeometry_descriptor_area50_512.so libgeometry_descriptor_area50_1024.so

libgeometry_descriptor_area50_512.so: geometry_descriptor.cpp
	$(CXX) $(CXXFLAGS) -shared -DIMAGE_HEIGHT=512 -DIMAGE_WIDTH=512 $< -o $@

libgeometry_descriptor_area50_1024.so: geometry_descriptor.cpp
	$(CXX) $(CXXFLAGS) -shared -DIMAGE_HEIGHT=1024 -DIMAGE_WIDTH=1024 $< -o $@

.PHONY: all
