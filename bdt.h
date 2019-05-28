//
// Created by felix on 03/12/18.
//

#ifndef HPC_BDT_BDT_H
#define HPC_BDT_BDT_H

#include <vector>
#include <malloc.h>
#include <tuple>
#include <omp.h>
#include <cfloat>
#include "structs.h"
#include "utility.h"
#include "backfitting.h"
#include "old_versions.h"


/**
 * moved things in heap instead of stack, because of memory (problems)
 */

// returns (gain, c) on that feature
// shift=1 when creating decision table, shift>1 when backfitting
std::tuple<long double, int> gain_on_feature(const int nr_leaves, const unsigned long N, const std::vector<int>& L,
                                        std::vector<float>& response,
                                        const std::vector<int>& sorted_feature,
                                        const std::vector<int>& run, const int shift) {
    int act, idx, best_idx=0;
    long double g=0, best_gain=-LDBL_MAX;
    std::vector<int> count(nr_leaves, 0);
    std::vector<double> summer(nr_leaves, 0);

    // 1 ms.
    for (int m = 0; m < N; ++m) {
        count[L[m]] += 1;
        summer[L[m]] += response[m];
    }

    auto doc_id = sorted_feature.begin();

    idx=0;

    // loop on nr of values of that feature - 30ms. most expensive part
    for (const int &r : run) {
        act=0;
        // update count and summer - most consuming
        // by changing position of docs having that value on feature j
        while(act!=r){
            auto ptr = *doc_id; // gained 8ms
            int lptr = L[ptr];
            count[lptr] -= 1;
            summer[lptr] -= response[ptr];
            count[lptr-shift] += 1;
            summer[lptr-shift] += response[ptr];
            ++doc_id;
            ++act;
        }
        idx += act;

        // calculate gain for that combination of (x_j, c)
        g = 0;
        for(int k=0; k<nr_leaves; k=k+2){
            if(count[k]>0)   g += std::pow(summer[k], 2) / count[k];
            if(count[k+1]>0) g += std::pow(summer[k+1], 2) / count[k+1];
        }

        if( g > best_gain ){
            best_gain = g;
            best_idx = idx;
        }
    } // end loop on nr of values of that feature

    std::vector<int>().swap(count);
    std::vector<double>().swap(summer);
    return std::make_tuple(best_gain, best_idx);
}

template<unsigned short d>
dt<d> create_dt(matrix& features, imatrix& sorted_features,
                imatrix& runs, std::vector<float>& response){
    unsigned long N = sorted_features[0].size();
    unsigned int n_feat = sorted_features.size();
    int best_feature=0;
    int nr_leaves=1, best_idx;
    long double best_gain;
    float best_c=0;

    std::vector<int> L(N, 1);

    dt<d> result_dt;

    // loop on all levels of the tree
    for(unsigned short t=0; t<d; ++t) {
        nr_leaves <<= 1; // number of leaves = 2^(t+1)
        best_gain=-LDBL_MAX, best_idx=0, best_feature=-1;

        // loop on features
        #pragma omp parallel if(par_dt<=par_value)
        {
            #pragma omp for schedule(dynamic)
            for(int j=0; j<n_feat; ++j){
                // loop on nr of values of that feature
                // returns a tuple with (gain, idx on sorted feature j)
                std::tuple<long double, int> gc =
                        gain_on_feature(nr_leaves, N, L, response, sorted_features[j], runs[j], 1);

                #pragma omp critical
                {
                    if (std::get<0>(gc) > best_gain) {
                        best_gain = std::get<0>(gc);
                        best_idx = std::get<1>(gc);
                        best_feature = j;
                    }
                };
            } // end loop on feature
        };

        // int mask = 0x00000001;
        auto doc_id = sorted_features[best_feature].begin();
        auto end = sorted_features[best_feature].end();
        auto best_point = doc_id+best_idx;

        if(t < d-1) {
            // update docs which have feature x_j > c
            while(doc_id!=best_point){
                auto ptr = *doc_id;
                L[ptr] -= 1;
                L[ptr] <<= 1;
                L[ptr] += 1;
                ++doc_id;
            }
            // update docs with feature x_j <= c
            while(doc_id!=end){
                auto ptr = *doc_id;
                L[ptr] <<= 1;
                L[ptr] += 1;
                ++doc_id;
            }
        }else{
            while(doc_id!=best_point){
                L[*doc_id] -= 1;
                ++doc_id;
            }
        }

        // update level t of decision table
        best_c = features[best_feature][*best_point];
        result_dt.fill_level(L, response, best_feature, best_c, t);
    }// end loop on level of tree

    for(int pass=0; pass<nr_backfitting_passes; ++pass) {
        backfitting_cyclic(&result_dt, features, sorted_features, runs, L, response);
        // backfitting_random(&result_dt, features, sorted_features, runs, L, response);
        // backfitting_greedy(&result_dt, features, sorted_features, runs, L, response);
    }

    return result_dt;
}

template<unsigned d, unsigned short t>
bdt_scoring<d, t>* train(matrix& training_set, matrix& transposed_features, imatrix& sorted_feats,
                     imatrix& runs, std::vector<float>& train_gt,
                     matrix& validation_set, std::vector<float>& validation_gt) {
    unsigned long train_size = training_set.size();
    unsigned long validation_size = validation_gt.size();
    bdt_scoring<d, t> *bdt_table;
    bdt_table = new bdt_scoring<d, t>();
    std::vector<float> residuals(train_size, 0);
    std::vector<float> prediction(train_size + validation_size, 0);
    int last_rmse_idx=0;
    float rmse = 0, best_rmse = 0;
    int best_nr_of_dt = 0;

    // Create the initial decision table
    // dt<d> initial_dt = old_create_dt<d>(transposed_features, sorted_feats, runs, train_gt);
    dt<d> initial_dt = create_dt<d>(transposed_features, sorted_feats, runs, train_gt);
    dt<d> decision_tab; // for later

    // add it to the model, compute predictions, compute residuals
    bdt_table->add_dt(initial_dt);

    #pragma omp parallel if (par_validation <= par_value)
    {
        #pragma omp for schedule(static) nowait
        for (int e = 0; e < train_size; ++e) {
            prediction[e] += shrink * initial_dt.predict(training_set[e]);
            residuals[e] = train_gt[e] - prediction[e];
        }

        // calculate rmse on validation set
        #pragma omp for schedule(static) reduction(+:rmse)
        for (int e = 0; e < validation_size; ++e) {
            prediction[e + train_size] += shrink * initial_dt.predict(validation_set[e]);
            float difference = validation_gt[e] - prediction[e + train_size];
            rmse += std::pow(difference, 2);
        }
    };

    rmse /= validation_size;
    rmse = std::sqrt(rmse);
    best_rmse = rmse;

    // procede for successive decision tables
    for (int tab = 1; tab < t; ++tab) {
        // update residuals
        decision_tab = create_dt<d>(transposed_features, sorted_feats, runs, residuals);
        bdt_table->add_dt(decision_tab);
        rmse = 0;

        #pragma omp parallel if(par_validation<=par_value)
        {
            #pragma omp for schedule(static) nowait
            for (int e = 0; e < train_size; ++e) {
                prediction[e] += shrink * decision_tab.predict(training_set[e]);
                residuals[e] = train_gt[e] - prediction[e];
            }

            // calculare rmse on validation set
            #pragma omp for schedule(static) reduction(+:rmse)
            for (int e = 0; e < validation_size; ++e) {
                prediction[e + train_size] += shrink * decision_tab.predict(validation_set[e]);
                float difference = validation_gt[e] - prediction[e + train_size];
                rmse += std::pow(difference, 2);
            }
        };

        rmse /= validation_size;
        rmse = std::sqrt(rmse);

        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_nr_of_dt = tab;
        }

        // check whether the actual rmse is the worse of the last recorded rmses
    }
    
    bdt_table->set_optimal_nr_tables(best_nr_of_dt);
    return bdt_table;
}

template<unsigned d, unsigned short t>
double test(const matrix &test_set, const std::vector<float> &ground_truth, bdt_scoring<d,t> *bdt_table,
        const float gt_mean, const float gt_std){
    unsigned long test_size = test_set.size();
    double rmse = 0;

    #pragma omp parallel if(par_test<=par_value)
    {
        #pragma omp for schedule(static) reduction(+:rmse)
        for (int i = 0; i < test_size; ++i) {
            float resp = bdt_table->predict(test_set[i]);
            rmse += std::pow(((resp - ground_truth[i]) - gt_mean) / gt_std, 2);
        }
    };
    rmse /= test_size;
    return std::sqrt(rmse);
}
#endif //HPC_BDT_BDT_H
