/*
 * 1024x1024 RAW file grayscale converter
 *
 * compile: nvcc hello_cuda.cu -o hello
 *
 *  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


__global__ void RGBtoGrayscale(unsigned char* grayImage, unsigned char* rgbImage, int width, int totalValues){

    // I used 1024 x 1024 blocks for simplicity
    int row = blockIdx.x;
    int col = blockIdx.y;
    
    // Offset for r of corresponding pixel
    int offset = (row*width + col) * 3;

    // Makes sure not to overrun data arrays
    if (offset < totalValues) {
        
        // assigning r, g, and b
        unsigned char r = rgbImage[offset];
        unsigned char g = rgbImage[1 + offset];
        unsigned char b = rgbImage[2 + offset];

        // assigning value to new gray
        unsigned char gray = (unsigned char)(0.21f*r + 0.71f*g + 0.07f*b);

        // saving gray image
        grayImage[offset] = gray;
        grayImage[offset + 1] = gray;
        if (offset > 3) { // avoids messing with a header in the .raw file. Makes it so image actually loads.
            grayImage[offset + 2] = gray;
        } else {
            grayImage[offset + 2] = b;
        }
    }
}

int main() {
    /*********************************************************************************/
    // Opening input file and loading it into memory
    printf("Opening input file\n");
    FILE* rgbImageFile = fopen("gc_conv_1024x1024.raw", "rb");
    if (!rgbImageFile) {
        printf("Did not open file gc_conv_1024x1024.raw");
        return 1;
    }
    printf("Closing input file\n");

    fseek(rgbImageFile, 0, SEEK_END);

    long filesize = ftell(rgbImageFile);

    rewind(rgbImageFile);

    // Input from binary file
    unsigned char* rgbImage = (unsigned char*) malloc(filesize);
    // Output for binary file
    unsigned char* grayImage = (unsigned char*) malloc(filesize);

    // Used for reading and writing to files
    long pixelRgb = filesize / sizeof(unsigned char);

    printf("Preparing to read input file\n");

    fread(rgbImage, sizeof(unsigned char), pixelRgb, rgbImageFile);

    printf("Finished Reading input file. filesize = %ld\n", filesize);

    fclose(rgbImageFile);



    /*************************************************************************************************/
    // Generating grayscale image with CUDA

    unsigned char* grayImage_d;
    unsigned char* rgbImage_d;

    cudaMalloc((void **) &grayImage_d, filesize);
    cudaMalloc((void**) &rgbImage_d, filesize);

    cudaMemcpy(rgbImage_d, rgbImage, filesize, cudaMemcpyHostToDevice);

    dim3 DimBlock(1024, 1024, 1);

    // Launch GPU kernel
    RGBtoGrayscale<<<DimBlock,1>>>(grayImage_d, rgbImage_d, 1024, pixelRgb);

    // cuda synch barrier
    cudaDeviceSynchronize();

    cudaMemcpy(grayImage, grayImage_d, filesize, cudaMemcpyDeviceToHost);

    cudaFree(grayImage_d);
    cudaFree(rgbImage_d);

    /**************************************************************/
    // Saving grayscale image


    FILE* testOutput = fopen("gray.raw", "wb");
    printf("Opening output file\n");
    if (!testOutput) {
        printf("Did not open...");
        return 1;
    }
    printf("Closing output file\n");


    fwrite(grayImage, sizeof(unsigned char), pixelRgb, testOutput);
    printf("Finished writing to output file\n");
    fclose(testOutput);

    free(rgbImage);
    free(grayImage);

    return 0;
}