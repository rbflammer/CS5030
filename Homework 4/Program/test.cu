/*
 * Hello world cuda
 *
 * compile: nvcc hello_cuda.cu -o hello
 *
 *  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__ void vec_add(int* a, int *b, int* out, int width, int n){
    // thread id of current block (on x axis)
    int row = threadIdx.x;
    int col = threadIdx.y;

    // TODO: Test using threadidx.y
    int offset = (row * width) + col;

    if (offset < n) {
        out[offset] = a[offset] + b[offset];
    }
    
}

int main() {
    int n = 100;

    int* a;
    int* b;
    int* c;

    int* a_d;
    int* b_d;
    int* c_d;

    dim3 DimBlock(10, 10, 1);

    a = (int*)(malloc(sizeof(int) * n));
    b = (int*)(malloc(sizeof(int) * n));
    c = (int*)(malloc(sizeof(int) * n));

    cudaMalloc(&a_d, sizeof(int) * n);
    cudaMalloc(&b_d, sizeof(int) * n);
    cudaMalloc(&c_d, sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(a_d, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, a, sizeof(int) * n, cudaMemcpyHostToDevice);


    // Launch GPU kernel
    vec_add<<<1,DimBlock>>>(a_d, b_d, c_d, 10, n);

    // cuda synch barrier
    cudaDeviceSynchronize();

    cudaMemcpy(c, c_d, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}