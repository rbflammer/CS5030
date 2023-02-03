/* Author: Bryce Flammer
 * A#: A02250893
 * 
 * parameters: <bin count> <min meas> <max meas> <data count> <OPTIONAL: show data>
 * 
 * @param bin count: (Integer) The number of bins to create
 * @param min meas: (Float) Minimum value of the data
 * @param max meas: (Float) Maximum value of the data
 * @param data count: (Integer) The number of data elements to create
 * @param show data: (Any) OPTIONAL: When a value is entered, it will show the generated data. When omitted, it will not show the generated data
*/
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    
    int processes; // Number of processes
    int myRank;    // My process rank

    // Input variables
    int binCount;
    float minMeas;
    float maxMeas;
    int dataCount;

    char wishToSeeData = 0; // 0 for no, 1 for yes

    // Initializing MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


    // Reading in arguments
    if (myRank == 0) {
        // Prevents program from crashing when too few arguments are given
        if (argc < 5) {
            printf("Invalid arguments. Valid parameter input: \"<bin count> <min meas> <max meas> <data count>\"");
            return 0;
        }

        // Command line arguments
        binCount =  atoi(argv[1]);
        minMeas = atof(argv[2]);
        maxMeas = atof(argv[3]);
        dataCount = atoi(argv[4]);

        if (argc == 6) {
            wishToSeeData = 1;
        }
    }

    // Sending arguments to all processes
    MPI_Bcast(&binCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&minMeas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxMeas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);


    float data[dataCount];      // array of all the data to be generated. Only populated by root.
    float localData[dataCount]; // array to store the data needed locally
    float* bins = (float*) malloc(sizeof(float) * binCount); // array containing the maximum value of every bin
    int* binCounts = (int*) malloc(sizeof(int) * binCount); // array of the bin counts for this process
    

    // Creating bins and populating data
    if (myRank == 0) {
        // Stores difference between the minimum and maximum value. Used for creating bins
        float minMaxDifference = maxMeas - minMeas;

        // Generating the data array
        srand(100);
        for (int i = 0; i < dataCount; i++) {
            data[i] = (((float)rand() / (float)RAND_MAX) * minMaxDifference) + minMeas;
        }

        // Initializing bin values
        float binStep = minMaxDifference / binCount;
        float currentBin = minMeas + binStep;
        for (int i = 0; i < binCount; i++) {
            bins[i] = currentBin;
            currentBin += binStep;
        }
    }

    // Sending bin labels to all processes
    MPI_Bcast(bins, binCount, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Initializing the return data array
    // Each process does this individually
    for (int i = 0; i < binCount; i++) {
        binCounts[i] = 0;
    }

    // Used for sending generated data
    int* offsets;
    int* sCounts = (int *)malloc(sizeof(int) * processes);

    // Preparing offsets and sendcounts for data to be sent
    if (myRank == 0) {
        offsets = (int *)malloc(sizeof(int) * processes);
        int extraData = dataCount % processes;
        int step = dataCount / processes;
        int currentOffset = 0;
        for (int i = 0; i < processes; i++) {
            offsets[i] = currentOffset;
            sCounts[i] = step;
            currentOffset += step;

            if (extraData > 0) {
                extraData--;
                currentOffset++;
                sCounts[i]++;
            }
        }
    }

    // Sending send counts to all processes
    MPI_Bcast(sCounts, processes, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Will print data if a fifth parameter was entered
    if (myRank == 0) {
        if (wishToSeeData == 1) {
            printf("\nData\n");
            for (int i = 0; i < dataCount; i++) {
                printf("%f ", data[i]);
            }
            printf("\n\n");
        }
    }

    // Sending data to all other processes
    MPI_Scatterv(data, sCounts, offsets, MPI_FLOAT, &localData, sCounts[myRank], MPI_FLOAT, 0, MPI_COMM_WORLD);


    // Inserting local data into appropriate bins
    for (int i = 0; i < sCounts[myRank]; i++) {
        for (int j = 0; j < binCount; j++) { // Steps through every bin
            if (binCount <= 1) { // If there is only one bin, then all data goes inside it
                binCounts[j]++;
                break;
            } else if (localData[i] <= bins[j]) { // If the data is less than the current bin, then increment that bins' count
                binCounts[j]++;
                break;
            }
        }
    }

    // Used to store total bin counts
    int finalCounts[binCount];
    
    // Reducing local bin counts to the total bin count
    MPI_Reduce(binCounts, finalCounts, binCount, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


    if (myRank == 0) {
        // Displaying bin maxes
        printf("bin_maxes = ");
        for (int i = 0; i < binCount; i++) {
            printf("%.3f ", bins[i]);
        }
        printf("\n");

        // Displaying final bin counts
        printf("bin_counts = ");
        for (int i = 0; i < binCount; i++) {
            printf("%d ", finalCounts[i]);
        }
        printf("\n");
    }

    // freeing allocated memory
    free(bins);
    free(binCounts);
    free(sCounts);
    if (myRank == 0) {
        free(offsets);
    }

    // Finishing MPI
    MPI_Finalize();

    return 0;
}