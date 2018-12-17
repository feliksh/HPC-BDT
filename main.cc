#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <random>

#include "structs.h"
#include "bdt.h"

#define chrono_now std::chrono::high_resolution_clock::now()
#define chrono_diff(b,e) std::chrono::duration_cast<std::chrono::milliseconds>(e-b)

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
                                             int* N, unsigned long* n_features, char sep){
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



int main(int argc, char* argv[]){
    // init chrono
    std::string parent = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/datasets/";
    std::string file = parent+"winequality-white.csv"; // sep=';'
    std::string file2 = parent+"cal_housing.data"; // too big values of response
    //std::string news = parent+"OnlineNewsPopularity.csv";
    std::string news = parent+"NewsModified.csv";
    std::string toy = parent+"toy2.csv";
    std::string gender = parent+"gender.csv";
    std::string ai_example = parent+"ai.data";
    int N=0;
    unsigned long n_features = 0;
    unsigned short const d=4;
    int const tables=200;

    std::vector<float> gt;

    std::vector<std::vector<float>> data = extract_data(&file2, &gt, &N, &n_features, ',');

    // take a random permutation of data
    shuffle_data(data, gt);

    int test_size = N/10;
    N = N-test_size;

    std::vector<std::vector<float>> training_set(data.begin(), data.begin()+N);
    std::vector<std::vector<float>> test_set(data.begin()+N, data.end());

    std::vector<float> train_gt(gt.begin(), gt.begin()+N);
    std::vector<float> test_gt(gt.begin()+N, gt.end());

    // create a more memory compact structure and cache friendly (transposed)
    std::vector<std::vector<float>> transposed_features(n_features, std::vector<float>(N, 0));
    transpose(training_set, transposed_features);

    // sort features
    std::vector<std::vector<int>> runs(n_features, std::vector<int>());
    std::vector<std::vector<int>> sorted_feats(n_features, std::vector<int>());
    auto sort_begin = std::chrono::high_resolution_clock::now();
    sort_features(transposed_features, runs, sorted_feats);
    auto sort_end = std::chrono::high_resolution_clock::now();
    auto sort_elapsed = chrono_diff(sort_begin, sort_end);

    // start learning step
    auto train_begin = chrono_now;
    bdt_scoring<d, tables> *bdt = train<d, tables>(training_set, transposed_features, sorted_feats, runs, train_gt);
    auto train_end = chrono_now;
    auto train_elapsed = chrono_diff(train_begin, train_end);

    // start testing step
    auto test_begin = chrono_now;
    double rmse = test<d,tables>(test_set, test_gt, bdt);
    auto test_end = chrono_now;
    auto test_elapsed = chrono_diff(test_begin, test_end);

    //std::cout << "Shot: " << rmse << "/" << test_size << " (" << rmse/test_size << ")\n";
    std::cout << "RMSE: " << rmse << "\n";
    std::cout << "Time sort:\t" << sort_elapsed.count() << "ms." << std::endl;
    std::cout << "Time train:\t" << train_elapsed.count() << "ms." << std::endl;
    std::cout << "Time test:\t" << test_elapsed.count() << "ms." << std::endl;

    delete bdt;

}
