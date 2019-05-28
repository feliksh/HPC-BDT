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
#include "utility.h"
#include "bdt.h"

int main(int argc, char* argv[]){
    /**
     * file "winequality":  11 features on   4'899 objects
     * file2 "cal_housing":  8 features on  20'640 objects
     * music "year_pred_m": 90 features on 515'345 objects
     */
    std::string parent = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/datasets/";
    std::string file = parent+"winequality-white.csv"; // sep=';'
    std::string file2 = parent+"cal_housing.data";
    std::string music = parent+"year_pred_modified.txt";
    int N=0;
    unsigned long n_features = 0;
    unsigned short const d=4;
    int const max_nr_tables=10;
    srand(23);
    par_value=3; // parallelization level (see report and utility header)
    std::vector<float> gt;
    float gt_mean;
    float gt_std;

    // Extract data
    std::vector<std::vector<float>> data = extract_data(&music, &gt, &N, &n_features, ',');

    // for not biased music dataset tests
    // int test_size = 51630;
    int test_size = (N / 10)*2;
    int train_size = N - test_size;
    int validation_size = (train_size/10)*2;
    train_size -= validation_size;

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    // take a random permutation of data
    shuffle_data(data, gt);

    // Split data based on proportions
    std::vector<std::vector<float>> training_set(data.begin(), data.begin() + train_size);
    std::vector<std::vector<float>> validation_set(data.begin() + train_size,
                                                   data.begin() + train_size + validation_size);
    std::vector<std::vector<float>> test_set(data.begin() + train_size + validation_size, data.end());
    std::vector<float> train_gt(gt.begin(), gt.begin() + train_size);
    std::vector<float> validation_gt(gt.begin() + train_size, gt.begin() + train_size + validation_size);
    std::vector<float> test_gt(gt.begin() + train_size + validation_size, gt.end());

    // create a more memory compact and cache friendly (transposed) structure
    std::vector<std::vector<float>> transposed_features(n_features, std::vector<float>(train_size, 0));
    transpose(training_set, transposed_features);

    // sort features
    std::vector<std::vector<int>> runs(n_features, std::vector<int>());
    std::vector<std::vector<int>> sorted_feats(n_features, std::vector<int>());
    sort_features(transposed_features, runs, sorted_feats);


    // Train
    auto train_begin = chrono_now;
    bdt_scoring<d, max_nr_tables> *bdt =
            train<d, max_nr_tables>(training_set, transposed_features, sorted_feats,
                                    runs, train_gt, validation_set, validation_gt);
    auto train_end = chrono_now;


    // Test
    gt_mean = calc_mean(train_gt);
    gt_std = calc_std(train_gt, gt_mean);

    auto test_begin = chrono_now;
    double rmse = test<d, max_nr_tables>(test_set, test_gt, bdt, gt_mean, gt_std);
    auto test_end = chrono_now;

    // Print time
    auto train_elapsed = chrono_diff(train_begin, train_end);
    auto test_elapsed = chrono_diff(test_begin, test_end);

    std::cout << std::endl;
    std::cout << "Time to train:\t" << train_elapsed.count() << "ms.\n";
    std::cout << "Time to test:\t" << test_elapsed.count() << "ms.\n";

    std::cout << std::endl;
    std::cout << "RMSE on " << bdt->nr_tables << " tables:\t" << rmse << "\n";

}
