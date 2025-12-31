# Makefile for q-polynomial enumeration

UNAME_S := $(shell uname -s)

# Clang config (for enumerate_triangles)
ifeq ($(UNAME_S),Darwin)
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    CXX_CLANG = clang++
    CXXFLAGS_CLANG = -O3 -march=native -mtune=native -std=c++17 -flto -funroll-loops \
                     -finline-functions -fomit-frame-pointer -ffast-math \
                     -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    LDFLAGS_CLANG = -flto -L$(LIBOMP_PREFIX)/lib -lomp
    # GCC config (for enumerate_trie - faster)
    GXX = $(shell ls /opt/homebrew/bin/g++-* 2>/dev/null | tail -1)
    CXXFLAGS_GCC = -O3 -mcpu=native -std=c++17 -flto -funroll-loops \
                   -finline-functions -fomit-frame-pointer -ffast-math -fopenmp
    LDFLAGS_GCC = -flto -fopenmp
else
    # Linux
    CXX_CLANG = clang++
    CXXFLAGS_CLANG = -O3 -march=native -std=c++17 -flto -fopenmp
    LDFLAGS_CLANG = -flto -fopenmp
    GXX = g++
    CXXFLAGS_GCC = -O3 -march=native -mtune=native -std=c++17 -flto -funroll-loops \
                   -finline-functions -fomit-frame-pointer -ffast-math -fopenmp
    LDFLAGS_GCC = -flto -fopenmp
endif

all: enumerate_triangles enumerate_prune enumerate_prune_inv enumerate_gpu

enumerate_triangles: enumerate_triangles.cpp
	@echo "Compiling enumerate_triangles.cpp (clang)..."
	$(CXX_CLANG) $(CXXFLAGS_CLANG) $< -o $@ $(LDFLAGS_CLANG)
	@echo "Done. Run with: ./$@ <max_n>"

enumerate_prune: enumerate_prune.cpp
	@echo "Compiling enumerate_prune.cpp (gcc)..."
	$(GXX) $(CXXFLAGS_GCC) $< -o $@ $(LDFLAGS_GCC)
	@echo "Done. Run with: ./$@ <n> [max_q]"

enumerate_prune_inv: enumerate_prune_inv.cpp
	@echo "Compiling enumerate_prune_inv.cpp (gcc, aggressive opts)..."
	$(GXX) $(CXXFLAGS_GCC) -fwhole-program $< -o $@ $(LDFLAGS_GCC)
	@echo "Done. Run with: ./$@ <n> [max_q] [inv_threshold]"

enumerate_gpu: enumerate_gpu.mm
	@echo "Compiling enumerate_gpu.mm (Metal)..."
	clang++ -O3 -march=native -flto -ffast-math -std=c++17 -framework Metal -framework Foundation $< -o $@
	@echo "Done. Run with: ./$@ <n> [max_q] [inv_threshold]"

clean:
	rm -f enumerate_triangles enumerate_prune enumerate_prune_inv enumerate_gpu *.dat

test: enumerate_triangles
	./enumerate_triangles 6

.PHONY: all clean test
