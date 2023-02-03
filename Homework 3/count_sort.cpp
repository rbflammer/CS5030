/* 
Author: Bryce Flammer
A#: A02250893

Answers to questions:
1. i, j, and count should be private. a, temp, and n should be public.

2. parallelizing memcpy would have no benefit. In this particular setup, temp will be equal accross all threads so parallelizing it 
at best would only result in redundent execution, and at worst the creation of race conditions. To avoid race conditions, the command
"# pragma omp critical" could be used. 

3. Written below
*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
using namespace std;

void count_sort(int[], int, int);

int main(int argc, char *argv[]) {
    int thread_count = strtol(argv[1], NULL, 10);   // Number of threads
    int data_count = stoi(argv[2], NULL, 10);       // Number of integers to be generated
    int a[data_count];                              // Array of integers for generated data


    // Generating random array of integers and printing to screen
    cout << "original: ";
    srand(100);
    for (int i = 0; i < data_count; i++) {
        a[i] = (rand() % data_count) + 1;
        cout << a[i] << " ";
    }
    cout << endl;

    // Sorting array
    count_sort(a, data_count, thread_count);

    // Printing sorted array
    cout << "sorted: ";
    for (int i = 0; i < data_count; i++) {
        cout << a[i] << " ";
    }
    cout << endl;

    return 0;
}


// Takes an array, the size of the array, and the number of threads to be used when sorting the array
// Stores the sorted array in the passed in array
void count_sort(int a[], int n, int thread_count) {
    int i, j, count;                            // i and j are counters. count is used for determining appropriate array position
    int* temp = (int *) malloc(n*sizeof(int));  // temporary array to store sorted elements

    // Starts thread_count threads with each taking a portion of the following for loop. Each thread will have a private i, j, and count
    // Each thread will have access to the shared a, n, and temp
#   pragma omp parallel for num_threads(thread_count) private(i, j, count) shared(a, n, temp)
    for (i = 0; i < n; i++) { // Stepping through each element in the data array
        count = 0;
        for (j = 0; j < n; j++) { // Comparing said element to every other element in the array
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        }

        // In this particular case, this does not need to be parallelized because each thread will access a unique temp[count], 
        // and a[i] will not be modified until after parallelization
        temp[count] = a[i]; 
    }

    // Copying temporary array into a and freeing temp
    memcpy(a, temp, n*sizeof(int)); 
    free(temp);
}
