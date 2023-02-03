/*
 * Hello world cuda
 *
 * compile: nvcc hello_cuda.cu -o hello
 *
 *  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// int CHANNELS = 3;

// __global__ void RGBtoGrayscale(unsigned char* grayImage, unsigned char* rgbImage, int width, int height){

//     int col = threadIdx.x + blockIdx.x * blockDim.x;
//     int row = threadIdx.y + blockIdx.y * blockDim.y;

//     if (col < width && row < height) {
//         int grayOffset = row * width + col;

//         int rgbOffset = grayOffset * CHANNELS;
//         unsigned char r = rgbImage[rgbOffset];
//         unsigned char g = rgbImage[1 + rgbOffset];
//         unsigned char b = rbfImage[2 + rgbOffset];

//         grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
//     }

//     printf("Ciao belli from core %d!\n", tid);
// }

int main() {

    // char[21] filename = "gc_conv_1024x1024.raw";

    printf("Opening file\n");
    FILE* rgbImageFile = fopen("gc_conv_1024x1024.raw", "rb");
    if (!rgbImageFile) {
        printf("Did not open...");
        return 1;
    }
    printf("Closing file\n");

    fseek(rgbImageFile, 0, SEEK_END);

    long filesize = ftell(rgbImageFile);


    rewind(rgbImageFile);

    unsigned char* rgbImage = (unsigned char*) malloc(filesize);
    unsigned char* grayscaleImage = (unsigned char*) malloc(filesize);

    long pixels = filesize / sizeof(unsigned char);

    printf("Preparing to read\n");

    fread(rgbImage, sizeof(unsigned char), pixels, rgbImageFile);

    printf("Finished Reading %ld\n", filesize);

    // for (int i = 0; i < pixels; i++) {
    //     printf("%d", rgbImage[i]);
    // }
    fclose(rgbImageFile);

    memcpy(grayscaleImage, rgbImage, filesize);

    for (int i = 0; i < pixels; i++) {
        if (i % 3 == 2){
            grayscaleImage[i] = 0;
        } if ( i % 3 == 0) {
            grayscaleImage[i] = 0;
        }
    }


    FILE* testOutput = fopen("test_output.raw", "wb");
    if (!testOutput) {
        printf("Did not open...");
        return 1;
    }
    printf("Closing file\n");

    // for (int i = 0; i < pixels; i++) {
        // fwrite((unsigned char*)rgbImage[i], sizeof(unsigned char), 1, testOutput);
    // }
    fwrite(grayscaleImage, sizeof(unsigned char), pixels, testOutput);
    printf("Finished Printing (Supposedly)\n");
    fclose(testOutput);

    // Launch GPU kernel
    // RGBtoGrayscale<<<1,1>>>();

    // cuda synch barrier
    // cudaDeviceSynchronize();

    return 0;
}