# C++ ядро fuga — сборка (без внешних зависимостей, g++ ≥ 12, C++20)
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra

all: fuga_train fuga_decode

fuga_train: train.cpp fuga_core.h
	$(CXX) $(CXXFLAGS) train.cpp -o fuga_train

fuga_decode: decode.cpp fuga_core.h
	$(CXX) $(CXXFLAGS) decode.cpp -o fuga_decode

clean:
	rm -f fuga_train fuga_decode *.o

.PHONY: all clean