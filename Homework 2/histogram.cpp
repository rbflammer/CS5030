/* Reasons why results might differ accross executions:
1. Race conditions i.e. two threads editing the same variable at the same time
2. Premature computation of sections such as trying to compute the final sum before all threads are finished completing
I did a couple things to solve these problems. 
For problem one, I only gave each thread access the data and output variables that are solely for them.
Continuing on one, the main thread will do the work of summing up the final total to avoid any other race conditions. 
Before the main thread completes its final sum, it waits for all other threads to finish their computation, this solves the second problem.
*/
#include <iostream>
#include <thread>
#include <iomanip>
#include <vector>
#include <time.h>
using namespace std;

int computeHistogram(int binCount, float* bins, int myDataCount, float* myData, int* returnData);


int main(int argc, char *argv[]) {

    // Prevents program from crashing when too few arguments are given
    if (argc < 6) {
        printf("Invalid arguments. Valid parameter input: \"<number of threads> <bin count> <min meas> <max meas> <data count>\"");
        return 0;
    }

    // Command line arguments
    int threadCount = stoi(argv[1]);
    int binCount = stoi(argv[2]);
    float minMeas = stof(argv[3]);
    float maxMeas = stof(argv[4]);
    int dataCount = stoi(argv[5]);

    // Stores differnce between the minimum and maximum value. Used for creating bins
    float minMaxDifference = maxMeas - minMeas;


    vector<thread> threads; // vector of all the threads in the program
    float data[dataCount];  // array of all the data to be generated
    float* bins = new float[binCount]; // array containing the maximum value of every bin
    int* returnData = new int[threadCount*binCount]; // array of all the return data of the program

    // Generating the data array
    srand(100);
    for (int i = 0; i < dataCount; i++) {
        data[i] = (rand() % (int(minMaxDifference) + 1)) + minMeas;
    }

    // Initializing the return data array
    for (int i = 0; i < threadCount * binCount; i++) {
        returnData[i] = 0;
    }

    // Initializing bin values
    float binStep = minMaxDifference / binCount;
    float currentBin = minMeas + binStep;
    for (int i = 0; i < binCount; i++) {
        bins[i] = currentBin;
        currentBin += binStep;
    }

    // Starting threads
    int dataStep = dataCount / threadCount;
    int myDataIndex = 0;
    int myDataCount = dataStep;
    int dataMod = dataCount % threadCount; // Used to balance workload when dataCount % threadCount != 0
    for (int i = 0; i < threadCount; i++) {
        threads.push_back(thread(computeHistogram, binCount, bins, myDataCount, data + myDataIndex, returnData + (i * binCount)));

        myDataIndex += myDataCount;
        myDataCount = dataStep;

        if (dataMod > 0) {
            myDataCount++;
            dataMod--; 
        }
    }

    // Initializing the final bin counts
    int finalCounts[binCount];
    for (int i = 0; i < binCount; i++) {
        finalCounts[i] = 0;
    }

    // Waiting for all threads to complete
    for (int i = 0; i < threadCount; i++) {
        threads[i].join();
    }

    // Populating final bin counts
    int countIndex = 0;
    for (int i = 0; i < threadCount * binCount; i++) {
        if (countIndex == binCount) {
            countIndex = 0;
        }
        finalCounts[countIndex] += returnData[i];
        countIndex++;
    }

    // Displaying bin maxes
    cout << "bin_maxes = ";
    for (int i = 0; i < binCount; i++) {
        cout << fixed << setprecision(3) << bins[i] << " ";
    }
    cout << endl;

    // Displaying final bin counts
    cout << "bin_counts = ";
    for (int i = 0; i < binCount; i++) {
        cout << finalCounts[i] << " ";
    }

    return 0;
}

// @param binCount : Number of bins
// @param bins : Array of bin maximums
// @param myDataCount : Number of data elements
// @param myData : Array of data elements
// @param returnData : Array of return data
// @return 0 for thread
int computeHistogram(int binCount, float* bins, int myDataCount, float* myData, int* returnData) {
    
    for (int i = 0; i < myDataCount; i++) { // Steps through every data element given to the thread

        bool found = false; // used to break from next loop

        for (int j = 0;!found && j < binCount; j++) { // Steps through every bin

            if (binCount <= 1) { // If there is only one bin, then all data goes inside it
                returnData[j]++;
                found = true;
            } else if (myData[i] <= bins[j]) { // If the data is less than the current bin, then increment that bins' count
                returnData[j]++;
                found = true;
            }
        }
    }
    return 0;
}