# For debugging, enable DEBUG and disable OPTIMIZE
DNWARN = -Wall -Wextra -Wodr -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable
THREADED = -fopenmp -D_GLIBCXX_PARALLEL
# DEBUG = -g -fsanitize=address -lasan -lubsan #undefined
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native

SRC_UTILS = $(wildcard src/utils/*.cpp)

main:
	mpicxx -g -std=c++14 $(DNWARN) $(OPTIMIZE) $(DEBUG) -DLBMW -DChosenTerm2Doc=BM25 -DSHARD_RADIX=5 \
	$(SRC_UTILS) src/main.cpp src/folly/Select64.cpp -g  -I src $(THREADED) \
	-I src/FastPFor/headers src/FastPFor/src/bitpacking.cpp src/FastPFor/src/bitpackingaligned.cpp \
	src/FastPFor/src/bitpackingunaligned.cpp src/FastPFor/src/simdbitpacking.cpp \
	-mssse3 -o bin/testcase \
	; echo "";
