#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>

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


// TODO check bob correctness
std::vector<std::vector<int>> sort_features(const float* data, short n_features, int N,
                                            std::vector<std::vector<int>>& runs){
    int count=0;
    std::vector<int> v(N);
    std::vector<std::vector<int>> result(n_features, std::vector<int>());
    for(short feat=0; feat<n_features; ++feat){
        std::iota(v.begin(), v.end(), 0);
        std::sort(v.begin(), v.end(), [&](int i, int j){ return data[(feat*N)+i]>data[(feat*N)+j]; });
        result[feat].assign(v.begin(), v.end());
        std::vector<int> bob;
        for(int i=0; i<N-1; ++i){
            if(data[(feat*N)+v[i]] == data[feat*N]+v[i+1])
                ++count;
            else {
                bob.push_back(count+1);
                count = 0;
            }
        }
        bob.push_back(count+1);
        runs[feat].assign(bob.begin(), bob.end());
    }
    return result;
}

void inverted_index(std::vector<std::vector<int>> &inv_index, std::vector<float> unique_vals,
                    const float* features, unsigned int N){
    int k=0;
    for(int idx=0; idx<N; ++idx){
        k=0;
        while(unique_vals[k] != features[idx]) ++k;
        inv_index[k].push_back(idx);
    }
}

template<unsigned short d>
dt<d> create_dt(float *features, unsigned short n_feat, std::vector<int>& response){
    unsigned int N = response.size(), uq_size=0, best_feature=0;
    short *L;
    int nr_leaves=1;
    std::vector<int>count;
    int tmp=0;
    float best_gain=0, c=0, g=0;
    std::vector<float> summer;
    std::vector<float> unique_vals;
    std::vector<std::vector<int>> inv_index;

    L=(short*)malloc(N*sizeof(short));
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
                unique_vals.push_back(features[(j*N) + tmp]);
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
            inv_index = std::vector<std::vector<int>>(uq_size, std::vector<int>(0));
            inverted_index(inv_index, unique_vals, features+(j*N), N);

            // allocate memory for 2^d entries for count and sum
            //count = (int*)malloc(nr_leaves*sizeof(int));
            //summer = (float*)malloc(nr_leaves*sizeof(float));
            // fill them in
            //memset(count, 0, nr_leaves*sizeof(int));
            //memset(summer, 0, nr_leaves*sizeof(float));
            for(int m=0; m<nr_leaves; ++m){
                count.push_back(0);
                summer.push_back(0);
            }
            for(int m=0; m<N; ++m){
                count[L[m]] += 1;
                summer[L[m]] += response[m];
            }
            /**if(j == 0) {
                std::cout << "level " << t << ":" << std::endl;
                for (int m = 0; m < nr_leaves; ++m)
                    std::cout << count[m] << " ";
                std::cout << std::endl;
            }**/

            // find best value which maximizes gain
            for(unsigned int idx=0; idx<uq_size-1; ++idx){
                // update table of count-sum with these index changes
                for(std::vector<int>::iterator it=inv_index[idx].begin(); it != inv_index[idx].end(); ++it){
                    count[L[*it]] -= 1;
                    summer[L[*it]] -= response[*it];
                    count[L[*it]-1] += 1;
                    summer[L[*it]-1] += response[*it];
                }
                inv_index[idx].clear();
                // for(int i=0; i<nr_leaves; ++i) std::cout << count[i] << " ";
                // std::cout << std::endl;
                // calculate gain for such division
                g = 0;
                for(int k=0; k<nr_leaves; ++k) {
                    if (count[k] != 0) {
                        g += (summer[k] * summer[k]) / count[k];
                    }
                }
                if( g > best_gain ){
                    best_gain = g;
                    best_feature = j;
                    c = unique_vals[idx+1];
                }
            }

            // free inverted index (automatically clears interior vectors)
            summer.clear();
            count.clear();
            unique_vals.clear();
            inv_index.clear();
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
        //free(count);
        //free(sum);
        // update L with new division
        if(t != d-1) {
            for (tmp = 0; tmp < N; ++tmp) {
                if (features[(best_feature * N) + tmp] > c)
                    L[tmp] -= 1;
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
    short n_features = 11;
    unsigned short const d=3;

    auto begin = std::chrono::high_resolution_clock::now();

    float* features = nullptr;
    std::vector<std::vector<float>> data = extract_data(file, &response, &N);

    // create a more memory compact structure and cache friendly (transposed)
    features = (float*)malloc(11*N*sizeof(float));
    transform(data, features, N, 11);

    std::vector<std::vector<int>> runs(n_features, std::vector<int>());
    std::vector<std::vector<int>> sorted_feats = sort_features(features, n_features, N, runs);

    for(int i=0; i<4; ++i){
        for(int j=0; j<20; ++j){
            std::cout << features[(i*N)+sorted_feats[i][j]] << "; ";
        }
        std::cout << "\n\n";
    }
    //dt<d> result = create_dt<d>(features, 11, response);

    dt<d> result = create_dt2<d>(sorted_feats)
    //result.printer();

    /**
    for(int i=0; i<20; ++i)
        std::cout << "prediction: " << result.predict(data, i, N)
                  << "\tground truth:" << response[i] << std::endl;
    **/

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin);

    std::cout << "Time:\t" << elapsed.count() << "ms." << std::endl;
}
