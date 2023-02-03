/*
 * Hello world cuda
 *
 * compile: nvcc hello_cuda.cu -o hello
 *
 *  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__ void cuda_hello(){
    // thread id of current block (on x axis)
    int tid = threadIdx.x;

    // block id (on x axis)
    //int bx = blockDim.x;

    printf("Ciao belli from core %d!\n", tid);
}

int main() {
    // Launch GPU kernel
    cuda_hello<<<1,1>>>();

    // cuda synch barrier
    cudaDeviceSynchronize();

    return 0;
}