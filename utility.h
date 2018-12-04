
#ifndef HPC_BDT_UTILITY_H
#define HPC_BDT_UTILITY_H

#include "bdt.h"

#define swape(x,y){temp=x;x=y;y=temp;}

void transpose(std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& transposed){
    for(int i=0; i<data.size(); ++i){
        for(int j=0; j<data[0].size(); ++j){
            transposed[j][i] = data[i][j];
        }
    }
}

// odd-even sort in sequential mode
void sequential_sort(std::vector<float>& data, unsigned int size){
    bool isSorted = false;
    float temp;
    while (!isSorted){
        isSorted = true;

        for (int i=1; i<=size-2; i=i+2){
            if (data[i] < data[i+1]){
                swape(data[i],data[i+1])
                isSorted = false;
            }
        }
        for (int i=0; i<=size-2; i=i+2){
            if (data[i] < data[i+1]){
                swape(data[i],data[i+1])
                isSorted = false;
            }
        }
    }
}



#endif //HPC_BDT_UTILITY_H
