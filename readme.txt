Sample: sortingNetworks
Minimum spec: SM 3.0

This sample implements bitonic sort and odd-even merge sort (also known as Batcher's sort), algorithms belonging to the class of sorting networks. While generally subefficient, for large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), this may be the preferred algorithms of choice for sorting batches of short-sized to mid-sized (key, value) array pairs. Refer to an excellent tutorial by H. W. Lang http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm

Key concepts:
Data-Parallel Algorithms

# How to run:
make - compiles the code using the nvcc compilter
make clean - cleans the built binaries
make gdb=1 - creates binaries with debug symbols loaded
make run k - runs the bitonic sorting. k goes from 0 to 26. k=20 by default



