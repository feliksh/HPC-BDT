#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "structs.h"
#include "utility.h"

std::vector<std::vector<float>> extract_data(const std::string filename, std::vector<int> *response, int* N){
    std::vector<std::vector<float>> features;
    std::vector<float> line_f;
    int i=0;
    std::string line;

    // 11 feat vects + 1 response
    std::ifstream file;
    file.open(filename);
    if(file.good())
        std::getline(file, line, '\n'); // header

    if(file.good()) {
        while (std::getline(file, line, ';')) {
            ++i;
            line_f.push_back(std::stof(line));
            if (i == 11) {
                features.push_back(line_f);
                line_f.clear();
                std::getline(file, line, '\n');
                response->push_back((int) std::stof(line));
                i = 0;
                *N = *N + 1;
            }
        }
    }
    file.close();
    return features;
}

void inverted_index(std::vector<std::vector<int>> inv_index, float* unique_vals, float* features, unsigned int N){
    short k=0;
    for(int idx=0; idx<N; ++idx){
        k=0;
        while(unique_vals[k] != features[idx]) ++k;
        inv_index[k].push_back(idx);
    }
}


void create_dt(float *features, unsigned short n_feat, std::vector<int> response, unsigned short d){
    unsigned int N = response.size(), uq_size=0;
    double nr_leaves = 1, tmpl=0;
    short L[N];
    float best_gain=0, c=0, unique_vals[N], g=0;
    int *nr_occ = nullptr, *count = nullptr;
    float *sum_resp = nullptr, *sum = nullptr;
    int tmp=0;
    memset(L, 0, N*sizeof(short));
    // for each level of d
    for(unsigned short t=0; t<d; ++t) {
        nr_leaves = std::pow(2, t);
        // for each feature x_j, find the one with highest gain
        for (short j = 0; j < n_feat; ++j) {
            while (tmp != N) {
                unique_vals[tmp] = L[(N * j) + tmp];
                ++tmp;
            }
            sequential_sort(unique_vals, N);
            // extract unique values
            uq_size = 0;
            tmp = 0;
            while (uq_size != N && tmp != N) {
                if (unique_vals[uq_size] != unique_vals[tmp]) {
                    ++uq_size;
                    unique_vals[uq_size] = unique_vals[tmp];
                }
                ++tmp;
            }
            // construct inverted index
            std::vector<std::vector<int>> inv_index(uq_size, std::vector<int>());
            // TODO check that &features[j*N] is correct
            inverted_index(inv_index, unique_vals, &features[j*N], N);

            // allocate memory for 2^d entries for count and sum
            count = (int *)alloca(nr_leaves*sizeof(int));
            sum = (float *)alloca(nr_leaves*sizeof(float));
            // fill them in
            for(unsigned int i=0; i<N; ++i){
                count[L[i]] += 1;
                sum[L[i]] += response[L[i]];
            }

            // find best value which maximizes gain
            for(unsigned int idx=0; idx<uq_size; ++idx){
                // update table of count-sum with these indexes changes
                for(std::vector<int>::iterator it=inv_index[idx].begin(); it != inv_index[idx].end(); ++it){
                    count[L[*it]] -= 1;
                    sum[L[*it]] -= response[L[*it]];
                    count[L[*it]-1] += 1;
                    sum[L[*it]-1] += response[L[*it]];
                }
                // calculate gain for such division
                g = 0;
                for(short k=0; k<nr_leaves; ++k)
                    if(count[k] != 0)
                        g += (sum[k]*sum[k])/count[k];
                if( g > best_gain ){
                    best_gain = g;
                    c = unique_vals[idx];
                }
            }

        }
    }
}


int main(int argc, char* argv[]){
    // init chrono
    std::vector<int> response;
    std::string file = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/winequality-white.csv";
    int N=0;
    float sum=0;
    std::vector<std::vector<float>> data = extract_data(file, &response, &N);

    float one_d[11*N];
    for(int j=0; j<11; ++j){
        for(int i=0; i<N; ++i){
            one_d[j*N+i] = data[i][j];
        }
    }
    auto begin = std::chrono::high_resolution_clock::now();
    float var_one[N];
    unsigned int i=0, j=0;
    while(i!=N) {
        var_one[i] = one_d[i];
        ++i;
    }

    sequential_sort(var_one, N);

    i=0;
    while(i!=N && j!=N){
        if(var_one[i]!=var_one[j]){
            ++i;
            var_one[i] = var_one[j];
        }
        ++j;
    }
    j=0;
    // while(j!=i)
        // std::cout << var_one[j++] << std::endl;

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin);

    std::cout << "Time:\t" << elapsed.count() << "ms." << std::endl;
}
