/* author: Bryce Flammer
 * A#: A02250893
 *
 * raw file grayscale converter
 *
 * compile: nvcc grayscaleConverter.cu -o grayscaleConverter
 * run: grayscaleConverter <FILE ADDRESS> <IMAGE WIDTH> <IMAGE HEIGHT>
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

int main(int argc, char* argv[]) {
    /*********************************************************************************/
    // Opening input file and loading it into memory

    if (argc < 4) {
        printf("Incorrect number of arguments. Correct parameters are:\n<FILE ADDRESS> <IMAGE WIDTH> <IMAGE HEIGHT>");
    }

    int width = atoi(argv[2]);
    int height = atoi(argv[3]);

    printf("Opening input file\n");
    FILE* rgbImageFile = fopen(argv[1], "rb");
    if (!rgbImageFile) {
        printf("Did not open file");
        return 1;
    }

    fseek(rgbImageFile, 0, SEEK_END);

    long filesize = ftell(rgbImageFile);

    rewind(rgbImageFile);

    // Input from binary file
    unsigned char* rgbImage = (unsigned char*) malloc(filesize);
    // Output for binary file
    unsigned char* grayImage = (unsigned char*) malloc(filesize);

    // Used for reading and writing to files
    long pixelRgb = filesize / sizeof(unsigned char);

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

    dim3 DimBlock(width, height, 1);

    // Launch GPU kernel
    RGBtoGrayscale<<<DimBlock,1>>>(grayImage_d, rgbImage_d, width, pixelRgb);

    // cuda synch barrier
    cudaDeviceSynchronize();

    cudaMemcpy(grayImage, grayImage_d, filesize, cudaMemcpyDeviceToHost);

    cudaFree(grayImage_d);
    cudaFree(rgbImage_d);

    /**************************************************************/
    // Saving grayscale image


    FILE* testOutput = fopen("output.raw", "wb");
    printf("Opening output file (output.raw)\n");
    if (!testOutput) {
        printf("Did not open output.raw");
        return 1;
    }

    fwrite(grayImage, sizeof(unsigned char), pixelRgb, testOutput);
    printf("Finished writing to output.raw. Write successful.\n");
    fclose(testOutput);

    free(rgbImage);
    free(grayImage);

    return 0;
}