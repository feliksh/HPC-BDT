
#ifndef HPC_BDT_UTILITY_H
#define HPC_BDT_UTILITY_H

void sequential_sort(float *data, unsigned int size){
    bool isSorted = false;
    float temp;
    while (!isSorted){
        isSorted = true;

        for (int i=1; i<=size-2; i=i+2){
            if (data[i] < data[i+1]){
                temp = data[i];
                data[i] = data[i+1];
                data[i+1] = temp;
                isSorted = false;
            }
        }
        for (int i=0; i<=size-2; i=i+2){
            if (data[i] < data[i+1]){
                temp = data[i];
                data[i] = data[i+1];
                data[i+1] = temp;
                isSorted = false;
            }
        }
    }
}



#endif //HPC_BDT_UTILITY_H
