NVCC      := $(shell which nvcc 2>/dev/null || echo /opt/cuda/bin/nvcc)
CXX       := g++
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra

TARGET    := stitch
SRCS      := src/main.cu src/v0_sequential.cpp src/stb_impl.cpp

all: $(TARGET)

$(TARGET): $(SRCS) src/common.h src/versions.h src/cuda_utils.cuh
	@if [ -x "$(NVCC)" ]; then \
		echo "Compiling with $(NVCC)..."; \
		$(NVCC) -O3 -std=c++17 $(SRCS) -o $(TARGET); \
	else \
		echo "nvcc not found, compiling CPU build with $(CXX)..."; \
		$(CXX) $(CXXFLAGS) -x c++ src/main.cu -x c++ src/v0_sequential.cpp src/stb_impl.cpp -o $(TARGET); \
	fi

cpu: $(SRCS) src/common.h src/versions.h src/cuda_utils.cuh
	$(CXX) $(CXXFLAGS) -x c++ src/main.cu -x c++ src/v0_sequential.cpp src/stb_impl.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
