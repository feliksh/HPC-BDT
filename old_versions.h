//
// Created by felix on 17/12/18.
//

#ifndef HPC_BDT_OLD_VERSIONS_H
#define HPC_BDT_OLD_VERSIONS_H

#include <vector>
#include <malloc.h>
#include <tuple>
#include <omp.h>
#include <cfloat>
#include "structs.h"
#include "utility.h"
#include "backfitting.h"


typedef std::vector<std::vector<float>> matrix;
typedef std::vector<std::vector<int>> imatrix;

/**
 * This version with for loop takes in average 1300 ms.
 * The actual version with while loop and ptrs  950 ms.
template<unsigned short d>
dt<d> old_create_dt(matrix& features, imatrix& sorted_features,
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


        int best_point;
        auto doc_ids = sorted_features[best_feature];

        // This was transformed in a while loop, using pointers.
        if(t < d-1) {
            #pragma omp parallel if(par_dt2<=par_value)
            {
                #pragma omp for schedule(static) nowait
                for (int e = 0; e < best_idx + 1; ++e) {
                    int id = doc_ids[e];
                    L[id] -= 1;
                    L[id] <<= 1;
                    L[id] += 1;
                }
                #pragma omp single
                best_point = doc_ids[best_idx];
                #pragma omp for schedule(static)
                for (int e = best_idx + 1; e < N; ++e) {
                    int id = doc_ids[e];
                    L[id] <<= 1;
                    L[id] += 1;
                }
            }
        }else{
            #pragma omp parallel if(par_dt2<=par_value)
            {
                #pragma omp for schedule(static)
                for (int e = 0; e < best_idx + 1; ++e) {
                    int id = doc_ids[e];
                    L[id] -= 1;
                }
            }
            best_point = doc_ids[best_idx];
        }

        // update level t of decision table
        best_c = features[best_feature][best_point];
        result_dt.fill_level(L, response, best_feature, best_c, t);

    }// end loop on level of tree

    for(int pass=0; pass<nr_backfitting_passes; ++pass) {
        backfitting_cyclic(&result_dt, features, sorted_features, runs, L, response);
        // backfitting_random(&result_dt, features, sorted_features, runs, L, response);
        // backfitting_greedy(&result_dt, features, sorted_features, runs, L, response);
    }
    return result_dt;
}
**/



/**
 * this version does not include a vector where predictions are stored and updated
 * as the number of decision tables grows, indeed it calculated the actual prediction
 * by calling the BDT, which means that for a high number of tables, it uselessly spends
 * some time to traverse all trees.

template<unsigned d, unsigned short t>
bdt_scoring<d, t>* train(matrix& training_set, matrix& transposed_features, imatrix& sorted_feats,
                         imatrix& runs, std::vector<float>& train_gt,
                         matrix& validation_set, std::vector<float>& validation_gt){
    unsigned long train_size = training_set.size();
    unsigned long validation_size = validation_gt.size();
    bdt_scoring<d, t> *bdt_table;
    bdt_table = new bdt_scoring<d,t>();
    std::vector<float> residuals(train_size, 0);
    float rmse=0, best_rmse=0;
    int best_nr_of_dt=0;

    // Create the initial decision table
    // dt<d> initial_dt = old_create_dt<d>(transposed_features, sorted_feats, runs, train_gt);
    dt<d> initial_dt = create_dt<d>(transposed_features, sorted_feats, runs, train_gt);

    // add it to the model and calculate residuals
    bdt_table->add_dt(initial_dt);
#pragma omp parallel for if(par_validation) schedule(static)
    for (int e = 0; e < train_size; ++e) {
        residuals[e] = train_gt[e] - bdt_table->predict(training_set[e]); // includes shrinkage
    }

    // calculate rmse on validation set
#pragma omp parallel for if(par_validation) schedule(static) reduction(+:rmse)
    for (int e = 0; e < validation_size; ++e) {
        float difference = validation_gt[e] - bdt_table->predict(validation_set[e]);
        rmse += std::pow(difference, 2);
    }

    rmse /= validation_size;
    rmse = std::sqrt(rmse);
    best_rmse = rmse;

    // procede for successive decision tables
    for (int tab = 1; tab < t; ++tab) {
        dt<d> decision_tab = create_dt<d>(transposed_features, sorted_feats, runs, residuals);
        bdt_table->add_dt(decision_tab);
        // update residuals

        ** old version 1 **
        #pragma omp parallel if(par_validation<=par_value)
        {
            #pragma omp for schedule(static)
            for (int e = 0; e < train_size; ++e) {
              residuals[e] = train_gt[e] - bdt_table->predict(training_set[e]);
            }
        }
        ** end old ver1 **


        auto par_begin = chrono_now;
        ** old version 2 **
        #pragma omp parallel if(par_validation)
        {
            std::vector<float> predictions(train_size);
            #pragma omp for schedule(static)
            for (int e = 0; e < train_size; ++e) {
                predictions[e] = bdt_table->predict(training_set[e]);
            }
            #pragma omp for schedule(static)
            for (int e = 0; e < train_size; ++e) {
                predictions[e] = train_gt[e] - predictions[e];
            }
        }
        ** end old ver2 **

        // calculare rmse on validation set
        rmse = 0;

        #pragma omp parallel for if(par_validation) schedule(static) reduction(+:rmse)
        for (int e = 0; e < validation_size; ++e) {
            float difference = validation_gt[e] - bdt_table->predict(validation_set[e]);
            rmse += std::pow(difference, 2);
        }

        rmse /= validation_size;
        rmse = std::sqrt(rmse);
        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_nr_of_dt = tab;
        }
    }


    bdt_table->set_optimal_nr_tables(best_nr_of_dt);
    return bdt_table;
}
**/

#endif //HPC_BDT_OLD_VERSIONS_H
