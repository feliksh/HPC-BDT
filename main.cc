#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>

#include "structs.h"
#include "bdt.h"

#define parallel_sort false

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


std::vector<std::vector<float>> extract_data(const std::string *filename, std::vector<float> *response,
                                             int* N, int* n_features, char sep){
    std::vector<std::vector<float>> features;
    std::vector<float> line_f;
    int i=-1;
    *n_features=0;
    std::string line;

    std::ifstream file;
    file.open(*filename);

    // read Header
    if(file.good()) {
        while (i != 0) {
            std::getline(file, line, '\n');
            for (char &it : line) if (it == sep) ++(*n_features);
            i = 0;
        }
        *n_features;
    }

    if(file.good()) {
        while (std::getline(file, line, sep)) {
            ++i;
            if(i!=*n_features) line_f.push_back(std::stof(line));
            else{
                line_f.push_back(std::stof(line));
                features.push_back(line_f);
                line_f.clear();
                std::getline(file, line, '\n');
                response->push_back(std::stof(line));
                i = 0;
                *N = *N + 1;
            }
        }
    }
    file.close();

    return features;
}


// TODO check bob correctness
std::vector<std::vector<int>> sort_features(std::vector<std::vector<float>>& data,
                                            std::vector<std::vector<int>>& runs){

    std::vector<int> bob;
    std::vector<int> v(data[0].size());
    std::vector<std::vector<int>> result(data.size(), std::vector<int>());
#pragma omp parallel if(parallel_sort) private(bob) firstprivate(v)
    {
#pragma for schedule(dynamic)
        for (int feat = 0; feat < data.size(); ++feat) {
            int count=0;
            std::iota(v.begin(), v.end(), 0);
            std::sort(v.begin(), v.end(), [&](int i, int j) { return data[feat][i] > data[feat][j]; });
            result[feat].assign(v.begin(), v.end());
            for (int i = 0; i < data[0].size() - 1; ++i) {
                if (data[feat][v[i]] == data[feat][v[i + 1]])
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
    return result;
}



int main(int argc, char* argv[]){
    // init chrono
    std::vector<float> gt;
    std::string parent = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/datasets/";
    std::string file = parent+"winequality-white.csv"; // sep=';'
    std::string file2 = parent+"cal_housing.data"; // too big values of response
    std::string news = parent+"OnlineNewsPopularity.csv";
    std::string toy = parent+"toy2.csv";
    std::string gender = parent+"gender.csv";
    std::string ai_example = parent+"ai.data";
    int N=0;
    int n_features = 0;
    unsigned short const d=4;
    int const tables=5;

    float* features = nullptr;

    std::vector<std::vector<float>> data = extract_data(&file2, &gt, &N, &n_features, ',');

    // TODO test what happens with greater range
    // scale response in range 0-1
    float min=FLT_MAX, max=-min;
    for(int i=0; i<N; ++i) {
        if (gt[i] < min) min = gt[i];
        if (gt[i] > max) max = gt[i];
    }
    for(int i=0; i<N; ++i) gt[i] = (gt[i]-min)/(max-min);

    int test_size = N/10;
    //N = N-test_size;

    std::vector<std::vector<float>> training_set =
            std::vector<std::vector<float>>(data.begin(), data.begin()+N);
    std::vector<std::vector<float>> test_set =
            std::vector<std::vector<float>>(data.begin()+N, data.end());
    std::vector<float> train_gt = std::vector<float>(gt.begin(), gt.begin()+N);
    std::vector<float> test_gt = std::vector<float>(gt.begin()+N, gt.end());

    // create a more memory compact structure and cache friendly (transposed)
    std::vector<std::vector<float>> transposed_features(n_features, std::vector<float>(N, 0));
    transpose(training_set, transposed_features);

    // sort features
    std::vector<std::vector<int>> runs(n_features, std::vector<int>());

    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> sorted_feats = sort_features(transposed_features, runs);

    // start learning step
    bdt_scoring<d, tables> bdt;
    bdt = train<d, tables>(training_set, sorted_feats, runs, train_gt);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    //float rmse = test<d>(test_set, test_gt, bdt);
    //std::cout << "RMSE: " << rmse << "\n";
    std::cout << "Time:\t" << elapsed.count() << "ms." << std::endl;
}
