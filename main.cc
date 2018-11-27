#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "structs.h"
#include "utility.h"

/**
 *
 * @param data pointer to vector of vectors
 * @param result pointer to memory where datas are going to end up
 * @param N nr of data
 * @param n_feat nr of features
 */
void transform(std::vector<std::vector<float>>& data, float* result, int N, int n_feat){
    for(int i=0; i<N; ++i)
        for(int j=0; j<n_feat; ++j)
            result[(j*N)+i] = data[i][j];
}


void extract_data(const std::string filename, std::vector<int> *response, int* N, float** result){
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

    // create a more memory compact structure and cache friendly (trasposed)

    *result = (float*)malloc(11*(*N)*sizeof(float));

    transform(features, *result, *N, 11);
    features.clear();
}

void inverted_index(std::vector<std::vector<int>> &inv_index, float* unique_vals, float* features, unsigned int N){
    int k=0;
    for(int idx=0; idx<N; ++idx){
        k=0;
        while(unique_vals[k] != features[idx]) ++k;
        (inv_index[k]).push_back(idx);
    }
}

template<unsigned short d>
dt<d> create_dt(float *features, unsigned short n_feat, std::vector<int>& response){
    unsigned int N = response.size(), uq_size=0, best_feature=0;
    unsigned short L[N];
    int nr_leaves=1;
    int *count = nullptr, tmp=0;
    float best_gain=0, c=0, unique_vals[N], g=0;
    float *sum = nullptr;

    for(int i=0; i<N; ++i) L[i] = 1;
    dt<d> result_dt;

    // for each level of d
    for(unsigned short t=0; t<d; ++t) {
        nr_leaves <<= 1; // 2^t
        // std::cout << "nr leaves: " << nr_leaves << std::endl;
        // std::cout << "L valuess: " << L[0] << " " << L[234] << " " << L[4200] << " " << L[2123] << std::endl;
        // for each feature x_j, find the one with highest gain
        for (unsigned int j = 0; j < n_feat; ++j) {
            tmp=0;
            while (tmp != N) {
                unique_vals[tmp] = features[(j*N) + tmp];
                ++tmp;
            }
            sequential_sort(unique_vals, N);

            // extract unique values
            uq_size = 0; tmp = 0;
            while (tmp != N) {
                if (unique_vals[uq_size] != unique_vals[tmp]) {
                    ++uq_size;
                    unique_vals[uq_size] = unique_vals[tmp];
                }
                ++tmp;
            }
            ++uq_size;

            // construct inverted index
            std::vector<std::vector<int>> inv_index(uq_size, std::vector<int>());
            inverted_index(inv_index, unique_vals, features+(j*N), N);

            // allocate memory for 2^d entries for count and sum
            count = (int *)malloc(nr_leaves*sizeof(int));
            sum = (float *)malloc(nr_leaves*sizeof(float));
            // fill them in
            memset(count, 0, nr_leaves*sizeof(int));
            memset(sum, 0, nr_leaves*sizeof(float));
            for(unsigned int i=0; i<N; ++i){
                count[L[i]] += 1;
                sum[L[i]] += response[i];
            }
            if(j == 0) {
                std::cout << "level " << t << ":" << std::endl;
                for (int m = 0; m < nr_leaves; ++m)
                    std::cout << count[m] << " ";
                std::cout << std::endl;
            }

            int counter = 0;
            // find best value which maximizes gain
            for(unsigned int idx=0; idx<uq_size-1; ++idx){
                counter = 0;
                // update table of count-sum with these index changes
                for(std::vector<int>::iterator it=inv_index[idx].begin(); it != inv_index[idx].end(); ++it){
                    count[L[*it]] -= 1;
                    sum[L[*it]] -= response[*it];
                    count[L[*it]-1] += 1;
                    sum[L[*it]-1] += response[*it];
                    ++counter;
                }
                // for(int i=0; i<nr_leaves; ++i) std::cout << count[i] << " ";
                // std::cout << std::endl;
                // calculate gain for such division
                g = 0;
                for(int k=0; k<nr_leaves; ++k) {
                    if (count[k] != 0) {
                        g += (sum[k] * sum[k]) / count[k];
                    }
                }
                if( g > best_gain ){
                    best_gain = g;
                    best_feature = j;
                    c = unique_vals[idx+1];
                }
            }

            // free inverted index (automatically clears interior vectors)
            for(std::vector<std::vector<int>>::iterator it = inv_index.begin(); it != inv_index.end(); ++it) {
                (*it).clear();
                std::vector<int>().swap(*it);
            }
            inv_index.clear();
            //std::vector<std::vector<int>>().swap(inv_index);
            //std::cout << "next feature \n";
        }
        /** here we have found the feature j which provides
         * the maximum gain among all other features
         * best_feat = j (index of best feature)
         * c = v_i (value of separation for feature j)
         * so we can update the decision table with such values
         */
        result_dt.fill_level(best_feature, c, t);
        // free structures for next iteration of t (next tree level)
        // unique_vals: will re-use the same portion of memory for t+1
        delete[] count;
        delete[] sum;
        // update L with new division
        if(t != d-1) {
            for (tmp = 0; tmp < N; ++tmp) {
                if (features[(best_feature * N) + tmp] > c) {
                    L[tmp] -= 1;
                }
                L[tmp] <<= 1;
            }
        }
        best_gain = 0;
    }

    result_dt.update_predictions(L, response, N);
    return result_dt;
}


int main(int argc, char* argv[]){
    // init chrono
    std::vector<int> response;
    std::string file = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/winequality-white.csv";
    int N=0;
    float sum=0;
    unsigned short n_features = 11;
    unsigned short const d=3;

    auto begin = std::chrono::high_resolution_clock::now();

    float* data = nullptr;
    extract_data(file, &response, &N, &data);

    dt<d> result = create_dt<d>(data, 11, response);

    result.printer();

    /**
    for(int i=0; i<20; ++i)
        std::cout << "prediction: " << result.predict(data, i, N)
                  << "\tground truth:" << response[i] << std::endl;
    **/

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin);

    delete[] data;
    std::cout << "Time:\t" << elapsed.count() << "ms." << std::endl;
}
