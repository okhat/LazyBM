#!/bin/bash
set -x
strategy=LazyMaxScore
index=crawl-ef

date=2020_Jul06

mkdir -p ~/"$date"/"$strategy"/"$index"/

for value in 7 # 6 5 8 9
do    
    for model in BM25 F2EXP PL2 SPL LMDir
    do
        mpicxx -g -std=c++14 -Wall -Wextra -Wodr -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable \
        src/utils/dist_timer.cpp src/utils/env.cpp src/utils/log.cpp src/main.cpp src/folly/Select64.cpp -g  -I src -fopenmp -D_GLIBCXX_PARALLEL \
        -DNDEBUG -O3 -flto -fwhole-program -march=native -DLazyMaxScore -DChosenTerm2Doc="$model" -DSHARD_RADIX="$value" \
        -o bin/lazy-maxscore \
        ; echo "";

        time mpirun -np 1  bin/lazy-maxscore  /datasets2/ClueWeb12_graph/CatB/1x/1.edges.bin 52343021 133248235 > ~/"$date"/"$strategy"/"$index"/"$model"-Block"$value".txt
    done
done

