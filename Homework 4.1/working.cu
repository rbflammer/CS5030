/* author: Bryce Flammer
 * A#: A02250893
 *
 * compile: nvcc transpose.cu -o transpose
 * run: transpose <FILE ADDRESS> <IMAGE WIDTH> <IMAGE HEIGHT>
 *
 * input file: .raw file to transpose. For example, gc_1024_1024.raw
 * output: transposed_image.raw
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cassert>

// Definitions of tile size to be used with tiling approach
const int TILE_HEIGHT = 10;
const int TILE_WIDTH = 30; // Must be 3 times TILE_HEIGHT

// Number of times to run each type of kernal transpose function
const int ITERATIONS = 200; 


// Function to serially transpose the image
// width and height are the total width and height of the image
// input is a pointer to the image to be transposed
// output is a pointer to the location to store the transposed image
void serialTranspose(unsigned char* input, int width, int height, unsigned char* output) {
    for (int row = 0; row < height; row++) {
        for (int column = 0; column < width * 3; column+=3) {
            int inputIndex = (row * width * 3) + column;
            int transposedIndex = (column * width) + row * 3;

            output[transposedIndex] = input[inputIndex];
            output[transposedIndex + 1] = input[inputIndex + 1];
            output[transposedIndex + 2] = input[inputIndex + 2];
        }
    }
}


// Kernal function to transpose the image using only global memory
// width is the width of the image
// input is a pointer to the image to be transposed
// output is a pointer to the location to save the output image
// totalValues is the total number of unsigned characters in the image 
__global__ void globalTranspose(unsigned char* input, unsigned char* output, int width, int totalValues){

    // I used width by height blocks for simplicity
    int row = blockIdx.x;
    int column = blockIdx.y;
    
    // Index for first value of pixel corresponding to this row and column
    int inputIndex = (row * width + column) * 3;

    // Index of transposed value corresponding to this row and column
    int transposedIndex = (column * width + row) * 3;

    // Makes sure not to overrun input image
    if (inputIndex < totalValues) {      
        // transposing image
        output[transposedIndex] = input[inputIndex];
        output[transposedIndex + 1] = input[inputIndex + 1];
        output[transposedIndex + 2] = input[inputIndex + 2];
    }
}


// Kernal function to transpose the image using tiling and shared memory
// width is the number of pixels wide the image is * 3
// height is the mumber of pixels high the image is
// input is a pointer to image to be transposed
// output is a pointer to the location to save transposed image
__global__ void tiledTranspose(unsigned char* input, unsigned char* output, int width, int height) {

    // TILE_WIDTH will always be a multiple of three, and 3*TILE_HEIGHT for a square image.
    int row = (blockIdx.y * TILE_HEIGHT) + threadIdx.y;
    int column = (blockIdx.x * TILE_WIDTH) + threadIdx.x;

    // The location of shared memory
    extern __shared__ unsigned char sharedMemory[];

    // Index for the unsigned character corresponding to this row and column
    int inputIndex = (row * width) + column;

    // Location to save input value in shared memory
    int sharedIndex = (threadIdx.y * TILE_WIDTH) + threadIdx.x;

    // Saving input value in shared memory if row and column are within the image
    if (row < height && column < width)
        sharedMemory[sharedIndex] = input[inputIndex];

    __syncthreads();

    // Index for the new location of the pixel in the output image (note, only used per pixel instead of per thread) 
    int transposeIndex = (((column / 3) * width) + row * 3);

    // Saving transposed pixel if pixel is within image
    if (row < height && column < width) {
        if(inputIndex % 3 == 0) {
            output[transposeIndex + 0] = sharedMemory[sharedIndex + 0];
            output[transposeIndex + 1] = sharedMemory[sharedIndex + 1];
            output[transposeIndex + 2] = sharedMemory[sharedIndex + 2];
        }
    }
}

int validate(unsigned char* input1, unsigned char* input2, int totalValues) {
    for (int i = 0; i < totalValues; i++) {
        if (input1[i] != input2[i]) {
            return i;
        }
    }
    return -1;
}

// Reports errors with CUDA
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


// Main function for program
int main(int argc, char* argv[]) {
    /*********************************************************************************/
    // Opening input file and loading it into memory

    if (argc < 4) {
        printf("Incorrect number of arguments. Correct parameters are:\n<FILE ADDRESS> <IMAGE WIDTH> <IMAGE HEIGHT>");
    }

    // Width and Height of image
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);

    // Opening input file
    FILE* rgbImageFile = fopen(argv[1], "rb");
    if (!rgbImageFile) {
        printf("Could not open input file.\n");
        return 1;
    }

    // Finding number of bytes in the file
    fseek(rgbImageFile, 0, SEEK_END);
    long filesize = ftell(rgbImageFile);
    rewind(rgbImageFile);

    // Input from binary file
    unsigned char* rgbImage = (unsigned char*) malloc(filesize);
    // Output for serial transpose binary file
    unsigned char* serialTransposedImage = (unsigned char*) malloc(filesize);
    // Output for serial transpose binary file
    unsigned char* globalTransposedImage = (unsigned char*) malloc(filesize);
    // Output for serial transpose binary file
    unsigned char* tiledTransposedImage = (unsigned char*) malloc(filesize);    

    // Used for reading and writing to files
    long pixelRgb = filesize / sizeof(unsigned char);

    // Reading file into memory
    fread(rgbImage, sizeof(unsigned char), pixelRgb, rgbImageFile);

    //Closing file
    fclose(rgbImageFile);


    /*********************************************************************************/
    // Performing serial transpose

    serialTranspose(rgbImage, width, height, serialTransposedImage);

    /*********************************************************************************/
    // Performing global transpose

    unsigned char* globalTransposedImage_d; // output for global transpose
    unsigned char* rgbImage_d;              // input for global and tiled transpose

    cudaMalloc((void **) &globalTransposedImage_d, filesize);
    cudaMalloc((void**) &rgbImage_d, filesize);

    // Copying input image to CUDA device
    cudaMemcpy(rgbImage_d, rgbImage, filesize, cudaMemcpyHostToDevice);

    // Dimension of grid used for global transpose
    dim3 DimThreads(width, height, 1);

    float averageMsGlobal;
    for (int i = 1; i < ITERATIONS; i++) {
        // Starting timing
        cudaEvent_t startGlobal, stopGlobal;
        checkCuda( cudaEventCreate(&startGlobal) );
        checkCuda( cudaEventCreate(&stopGlobal) );
        float msGlobal;
        checkCuda( cudaEventRecord(startGlobal, 0) );

        // Launch GPU kernel
        globalTranspose<<<DimThreads,1>>>(rgbImage_d, globalTransposedImage_d, width, pixelRgb);

        // cuda synch barrier
        cudaDeviceSynchronize();

        // Stopping timing
        checkCuda( cudaEventRecord(stopGlobal, 0) );
        checkCuda( cudaEventSynchronize(stopGlobal) );
        checkCuda( cudaEventElapsedTime(&msGlobal, startGlobal, stopGlobal) );
        if (i == 1) {
            averageMsGlobal = msGlobal;
        } else {
            averageMsGlobal += (msGlobal - averageMsGlobal) / i;
        }
    }

    // Copying output from device to host
    cudaMemcpy(globalTransposedImage, globalTransposedImage_d, filesize, cudaMemcpyDeviceToHost);
    cudaFree(globalTransposedImage_d);

    /*********************************************************************************/
    // validating global transpose

    int validationResult = validate(serialTransposedImage, globalTransposedImage, pixelRgb);
    if (validationResult == -1) {
        printf("Serial output equals output created using global memory\n");
    } else {
        printf("Serial output does not equal output created using global memory at position %d\n", validationResult);
    }


    /*********************************************************************************/
    // Performing tiled transpose

    unsigned char* tiledTransposedImage_d; // Output for tile transposing

    cudaMalloc((void **) &tiledTransposedImage_d, filesize);

    // Dimension of grid for tile transposing
    dim3 DimGrid(ceil((width * 3) / (1.0 * TILE_WIDTH)), ceil(height / (1.0 * TILE_HEIGHT)), 1);

    // Dimension of each block for tile transposing
    dim3 DimBlock(TILE_WIDTH, TILE_HEIGHT, 1);

    // Memory needed for each block of tile transposing
    size_t tiledMemory = sizeof(unsigned char) * TILE_WIDTH * TILE_HEIGHT;

    float averageMsTiled;
    for (int i = 1; i < ITERATIONS; i++) {
        // Starting timing
        cudaEvent_t startTiled, stopTiled;
        checkCuda( cudaEventCreate(&startTiled) );
        checkCuda( cudaEventCreate(&stopTiled) );
        float msTiled;
        checkCuda( cudaEventRecord(startTiled, 0) );

        // Launch GPU kernel
        tiledTranspose<<<DimGrid, DimBlock, tiledMemory>>>(rgbImage_d, tiledTransposedImage_d, width * 3, height);

        // cuda synch barrier
        cudaDeviceSynchronize();

        // Stopping timing
        checkCuda( cudaEventRecord(stopTiled, 0) );
        checkCuda( cudaEventSynchronize(stopTiled) );
        checkCuda( cudaEventElapsedTime(&msTiled, startTiled, stopTiled) );

        if (i == 1) {
            averageMsTiled = msTiled;
        } else {
            averageMsTiled += (msTiled - averageMsTiled) / i;
        }
    }



    // Copying output from device to host
    cudaMemcpy(tiledTransposedImage, tiledTransposedImage_d, filesize, cudaMemcpyDeviceToHost);

    cudaFree(tiledTransposedImage_d);
    cudaFree(rgbImage_d);

    /*********************************************************************************/
    // validating tiled transpose

    validationResult = validate(serialTransposedImage, tiledTransposedImage, pixelRgb);
    if (validationResult == -1) {
        printf("Serial output equals output created using shared memory (tiling)\n");
    } else {
        printf("Serial output does not equal output created using shared memory (tiling) at position %d\n", validationResult);
    }


    /*********************************************************************************/
    // Computing bandwidth

    printf("Average Global runtime: %f\n", averageMsGlobal);
    printf("Average Tiled runtime: %f\n", averageMsTiled);

    // To compute bandwidth, I used the equation: Effective Bandwidth (GB) = (Bytes Read + Bytes Written) / (time_seconds * 10^9)
    // Each transpose needs to read and write (width * height * 3) unsigned chars, which are 1 byte
    // To convert milliseconds to to seconds, it needs to be divided by 1000
    // Therefore, the denominator equals time_milliseconds * 10^6
    // Therefore, the formula for effective bandwidth is: (width * height * 3 * 2) / (t_ms * 10^6)

    printf("Effective Global Bandwidth (GB/s): %f\n", (width * height * 6) / (averageMsGlobal * 1e6)); // When I ran it, I got 0.392 GB/s
    printf("Effective Tiled Bandwidth (GB/s):  %f\n", (width * height * 6) / (averageMsTiled  * 1e6)); // When I ran it, I got 2.698 GB/s


    /**************************************************************/
    // Saving transposed image

    FILE* outputFile = fopen("transposed_image.raw", "wb");
    if (!outputFile) {
        printf("Did not open transposed_image.raw");
        return 1;
    }

    fwrite(tiledTransposedImage, sizeof(unsigned char), pixelRgb, outputFile);
    fclose(outputFile);
    printf("Finished writing transposed image created using tiling to transposed_image.raw.\n");

    free(rgbImage);
    free(serialTransposedImage);
    free(globalTransposedImage);
    free(tiledTransposedImage);

    return 0;
}