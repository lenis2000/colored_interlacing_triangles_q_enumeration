# Makefile for q-polynomial enumeration (Fenwick tree optimized CPU version)

# Detect Homebrew OpenMP on macOS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
    ifneq ($(BREW_PREFIX),)
        LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
        ifneq ($(LIBOMP_PREFIX),)
            CXX = clang++
            CXXFLAGS = -O3 -march=native -mtune=native -std=c++17 -flto -funroll-loops \
                       -finline-functions -fomit-frame-pointer -ffast-math \
                       -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
            LDFLAGS = -flto -L$(LIBOMP_PREFIX)/lib -lomp
        else
            $(warning libomp not found. Install with: brew install libomp)
            CXX = clang++
            CXXFLAGS = -O3 -march=native -std=c++17 -Wall
            LDFLAGS =
        endif
    else
        CXX = clang++
        CXXFLAGS = -O3 -march=native -std=c++17 -Wall
        LDFLAGS =
    endif
else
    # Linux
    CXX = g++
    CXXFLAGS = -O3 -march=native -mtune=native -std=c++17 -flto -funroll-loops \
               -finline-functions -fomit-frame-pointer -ffast-math -fopenmp
    LDFLAGS = -flto -fopenmp
endif

TARGET = enumerate_triangles
SOURCE = enumerate_triangles.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling $(SOURCE)..."
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)
	@echo "Done. Run with: ./$(TARGET) <max_n>"

clean:
	rm -f $(TARGET) canonical_n*.dat output_n*.txt

test: $(TARGET)
	./$(TARGET) 6

.PHONY: all clean test
