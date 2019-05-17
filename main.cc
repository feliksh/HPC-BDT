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
    /**
     * file "winequality":  11 features on   4'899 objects
     * file2 "cal_housing":  8 features on  20'640 objects
     * news "NewsModified": 59 features on  39'645 objects
     * music "year_pred_m": 90 features on 515'345 objects
     * toy   "toy2":         5 features on      60 objects
     * gender "gender":      2 features on     208 objects
     * ai_example "ai":      3 features on      10 objects
     */
    // init chrono
    std::string parent = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/datasets/";
    std::string file = parent+"winequality-white.csv"; // sep=';' TODO: change separator
    std::string file2 = parent+"cal_housing.data"; // too big values of response
    //std::string news = parent+"OnlineNewsPopularity.csv";
    std::string news = parent+"NewsModified.csv";
    std::string music = parent+"year_pred_modified.txt";
    std::string toy = parent+"toy2.csv";
    std::string gender = parent+"gender.csv";
    std::string ai_example = parent+"ai.data";
    int N=0, n_runs=1;
    unsigned long n_features = 0;
    unsigned short const d=4;
    int const tables=100;
    srand(23);

    std::vector <std::chrono::milliseconds::rep> sort_times(n_runs);
    double sum_rmse = 0;

    std::vector<float> gt;

    std::vector<std::vector<float>> data = extract_data(&music, &gt, &N, &n_features, ',');

    // TODO: only for train set
    float gt_mean;
    float gt_std;

    int test_size = (N / 10)*2;
    // for music dataset
    // int test_size = 51630;
    int train_size = N - test_size;
    int validation_size = (train_size/10)*2;
    train_size -= validation_size;

    for(int i=0; i<n_runs; ++i) {
        // take a random permutation of data
        shuffle_data(data, gt);

        std::vector<std::vector<float>> training_set(data.begin(), data.begin() + train_size);
        std::vector<std::vector<float>> validation_set(data.begin()+train_size,
                data.begin()+train_size+validation_size);
        std::vector<std::vector<float>> test_set(data.begin()+train_size+validation_size, data.end());

        std::vector<float> train_gt(gt.begin(), gt.begin() + train_size);
        std::vector<float> validation_gt(gt.begin()+train_size, gt.begin()+train_size+validation_size);
        std::vector<float> test_gt(gt.begin()+train_size+validation_size, gt.end());

        gt_mean = calc_mean(train_gt);
        gt_std = calc_std(train_gt, gt_mean);

        // create a more memory compact structure and cache friendly (transposed)
        std::vector<std::vector<float>> transposed_features(n_features, std::vector<float>(train_size, 0));
        transpose(training_set, transposed_features);

        // sort features
        std::vector<std::vector<int>> runs(n_features, std::vector<int>());
        std::vector<std::vector<int>> sorted_feats(n_features, std::vector<int>());
        auto sort_begin = std::chrono::high_resolution_clock::now();
        sort_features(transposed_features, runs, sorted_feats);
        auto sort_end = std::chrono::high_resolution_clock::now();
        auto sort_elapsed = chrono_diff(sort_begin, sort_end);
        // std::cout << "Time sort:\t" << sort_elapsed.count() << "ms." << std::endl;

        // start learning step
        auto train_begin = chrono_now;
        bdt_scoring<d, tables> *bdt = train<d, tables>(training_set, transposed_features, sorted_feats,
                runs, train_gt, validation_set, validation_gt);
        auto train_end = chrono_now;
        auto train_elapsed = chrono_diff(train_begin, train_end);
        // std::cout << "Time train:\t" << train_elapsed.count() << "ms." << std::endl;

        sort_times[i] = train_elapsed.count();

        // start testing step
        auto test_begin = chrono_now;
        double rmse = test<d, tables>(test_set, test_gt, bdt, gt_mean, gt_std);
        auto test_end = chrono_now;
        auto test_elapsed = chrono_diff(test_begin, test_end);
        // std::cout << "Time test:\t" << test_elapsed.count() << "ms." << std::endl;

        //std::cout << "Shot: " << rmse << "/" << test_size << " (" << rmse/test_size << ")\n";
        //std::cout << "RMSE: " << rmse << "\n";
        sum_rmse += rmse;

        //delete bdt;
    }
    auto mean_time = sort_times[0];
    for(int i=1; i<n_runs; ++i)
        mean_time += sort_times[i];
    mean_time /= n_runs;
    std::cout << "\nAvg. Time:\t" << mean_time << "ms.\n";
    std::cout << "Avg. RMSE: " << sum_rmse/n_runs << "\n";

}
