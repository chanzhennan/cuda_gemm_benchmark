rm -rf build
mkdir build && cd build
cd build
cmake -DCUDA_ARCH=80 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 ..
make -j
./cuda_benchmark
