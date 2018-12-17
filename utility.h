
#ifndef HPC_BDT_UTILITY_H
#define HPC_BDT_UTILITY_H

#include "bdt.h"

#define swape(x,y){temp=x;x=y;y=temp;}
#define swtemp(x,y,z){temp=x;x=y;y=temp;}

#define parallel_sort true
#define par_dt true
#define par_backfitting true

void transpose(std::vector<std::vector<float>> &data, std::vector<std::vector<float>> &transposed){
    for(int i=0; i<data.size(); ++i){
        for(int j=0; j<data[0].size(); ++j){
            transposed[j][i] = data[i][j];
        }
    }
}

void shuffle_data(std::vector<std::vector<float>> &data, std::vector<float> &gt){
    unsigned long n = data.size();
    std::vector<float> temp;
    float ftemp;
    while(n != 1){
        int k = static_cast<int>(rand()%n);
        --n;
        std::swap(gt[k],gt[n]);
        swtemp(data[k],data[n],temp);
    }
}


// odd-even sort in sequential mode
void sequential_sort(std::vector<float>* data, unsigned int size){
    bool isSorted = false;
    float temp;
    while (!isSorted){
        isSorted = true;

        for (int i=1; i<=size-2; i=i+2){
            if ((*data)[i] < (*data)[i+1]){
                swape((*data)[i],(*data)[i+1])
                isSorted = false;
            }
        }
        for (int i=0; i<=size-2; i=i+2){
            if ((*data)[i] < (*data)[i+1]){
                swape((*data)[i],(*data)[i+1])
                isSorted = false;
            }
        }
    }
}


// TODO check bob correctness
void sort_features(std::vector<std::vector<float>>& data,
                   std::vector<std::vector<int>>& runs,
                   std::vector<std::vector<int>>& sorted){

    std::vector<int> bob;
#pragma omp parallel if(parallel_sort) private(bob)
    {
#pragma for schedule(dynamic)
        for (int feat = 0; feat < data.size(); ++feat) {
            int count=0;
            std::vector<int> v(data[0].size());
            std::iota(v.begin(), v.end(), 0);
            std::sort(v.begin(), v.end(), [&](int i, int j) { return data[feat][i] > data[feat][j]; });
            sorted[feat].assign(v.begin(), v.end());
            for (int i = 0; i < data[0].size() - 1; ++i) {
                if (data[feat][v[i]] == data[feat][v[i+1]])
                    ++count;
                else {
                    bob.push_back(count + 1);
                    count = 0;
                }
            }
            bob.push_back(count + 1);
            runs[feat].assign(bob.begin(), bob.end());
            bob.clear();
        }
    };
}

#endif //HPC_BDT_UTILITY_H
