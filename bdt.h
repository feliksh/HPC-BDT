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

std::chrono::milliseconds::rep time_holder(0);
int time_counter(0);

/**
 * static
 * const
 * moved things in heap instead of stack, because of memory (problems)
 */

// TODO: takes 73 ms. in avg. but it is the most time consuming since it's called often
/**
 * it's better to improve this function in sequential fashion, since it makes not much sense
 * to have parallelization inside, it's better to have parallelization outside.
 *
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

// TODO check static-ness
/**
 * this function is called outside of the parallel part, thus it doesn't cause any nested loop.
 * it makes sense to parallelize as much as possible
 */
 /**
// Cyclic backfitting
template<unsigned short d>
void backfitting_cyclic(dt<d>* dt, const matrix& features, const imatrix& sorted_features,
                  const imatrix& runs, std::vector<int>& L, std::vector<float> &response){
    unsigned long N = sorted_features[0].size();
    unsigned long n_feat = sorted_features.size();
    long double best_gain=-LDBL_MAX;
    int best_idx=0, best_feature=-1;

    for(int t=0; t<d; ++t){
        best_gain=-LDBL_MAX;
        // remove the cut
        for(int m=0; m<N; ++m) L[m] |= (1<<(d-t-1));
        // loop on features
        #pragma omp parallel if(par_backfitting) shared(N,L,response,sorted_features,runs,t)
        {
            #pragma omp for schedule(dynamic)
            for(int j=0; j<n_feat; ++j){
                // loop on nr of values of that feature
                // returns a tuple with (gain, idx on sorted feature j)
                // TODO: check gain on t
                std::tuple<long double, int> gc =
                        gain_on_feature(1<<d, N, L, response, sorted_features[j], runs[j], 1<<(d-t-1));
                // TODO: critical section?
                if(std::get<0>(gc) > best_gain){
                    best_gain = std::get<0>(gc);
                    best_idx = std::get<1>(gc);
                    best_feature = j;
                }
            } // end loop on feature
        };
        auto doc_id = sorted_features[best_feature].begin();
        //auto end = sorted_features[best_feature].end();
        auto best_point = doc_id+best_idx;
        // update docs which have feature x_j > c
        while(doc_id!=best_point){
            L[*doc_id] -= (1<<(d-t-1));
            ++doc_id;
        }
        float best_c = features[best_feature][*best_point];
        (*dt).features[t] = best_feature;
        (*dt).cuts[t] = best_c;
    }
    (*dt).update_predictions(L, response, d-1);
}
**/
template<unsigned short d>
dt<d> create_dt(matrix& features, imatrix& sorted_features,
                imatrix& runs, std::vector<float>& response){
    unsigned long N = sorted_features[0].size();
    unsigned int n_feat = sorted_features.size();
    int best_feature=0;
    int nr_leaves=1, best_idx;
    long double best_gain;
    float best_c=0;

    // TODO: parallelize? lol
    std::vector<int> L(N, 1);

    dt<d> result_dt;

    // loop on all levels of the tree
    for(unsigned short t=0; t<d; ++t) {
        nr_leaves <<= 1; // number of leaves = 2^(t+1)
        best_gain=-LDBL_MAX, best_idx=0, best_feature=-1;

        // loop on features
        #pragma omp parallel if(par_dt)
        {
            #pragma omp for schedule(dynamic)
            for(int j=0; j<n_feat; ++j){
                // loop on nr of values of that feature
                // returns a tuple with (gain, idx on sorted feature j)
                // auto sort_begin = std::chrono::high_resolution_clock::now();
                std::tuple<long double, int> gc =
                        gain_on_feature(nr_leaves, N, L, response, sorted_features[j], runs[j], 1);
                // auto sort_end = std::chrono::high_resolution_clock::now();
                // auto sort_elapsed = chrono_diff(sort_begin, sort_end);
                // sum_times += sort_elapsed.count();

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

        // sumsum_times += sum_times/n_feat;

        // int mask = 0x00000001;
        auto doc_id = sorted_features[best_feature].begin();
        auto end = sorted_features[best_feature].end();
        auto best_point = doc_id+best_idx;

        // TODO: can't be better because there are random accesses to L, no caching
        // TODO: try to unroll it, make 4 calls, pass one to each thread
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
        // best_c = features[best_feature][sorted_features[best_feature][best_idx]];
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
    float rmse = 0, best_rmse = 0;
    int best_nr_of_dt = 0;

    // TODO: add criterion to stop nr of tables
    // Create the initial decision table
    // dt<d> initial_dt = old_create_dt<d>(transposed_features, sorted_feats, runs, train_gt);
    dt<d> initial_dt = create_dt<d>(transposed_features, sorted_feats, runs, train_gt);
    dt<d> decision_tab; // for later

    // add it to the model, compute predictions, compute residuals
    bdt_table->add_dt(initial_dt);

    // auto par_begin = chrono_now;
    #pragma omp parallel if (par_validation)
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
    }
    // auto par_end = chrono_now;
    // auto par_diff = chrono_diff(par_begin, par_end);
    // std::cout << "time for par validation:\t" << par_diff.count() << "ms.\n";
    // time_holder += par_diff.count();
    // ++time_counter;

    rmse /= validation_size;
    rmse = std::sqrt(rmse);
    best_rmse = rmse;
    // std::cout << "RMSE at lv 0: " << rmse << std::endl;
    // initial_dt.printer();

    // procede for successive decision tables
    // TODO: make predict a vector and update as nr of tables grows
    // TODO: make these vectors const
    // TODO: ordered

    for (int tab = 1; tab < t; ++tab) {
        // update residuals
        /** old version 1 **
        for (int e = 0; e < train_size; ++e) {
            residuals[e] = train_gt[e] - bdt_table->predict(training_set[e]);
        }
        ** end old ver1 **/

        /** old version 2 ** // There is a little improvement
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
        ** end old ver2 **/

        // TODO: replace call residual=gt-prediction, using fact that gt is fixed
        // TODO: oh shit, 12x improvement
        // TODO: that's big (even though it's only -120ms)
        /**
         * probably due to the fact that the compiler was not able to understand that
         * the call to bdt->predict could have been seen as a single value.
         * it had to call predict and create the environment for that call, and thus
         * loosing a lot of time before having a final value. while in this version
         * the calculation is trivial and the call to predict function of the dt is
         * probably inlineable (can try to specify inline keyword).
         * TODO: can try to make 2 for loops, one for predict[e] and one for resid[e]
         */
        // auto par_begin = chrono_now;

        // there is parallelism inside create_dt, so it's better not having this call inside pragma
        decision_tab = create_dt<d>(transposed_features, sorted_feats, runs, residuals);
        bdt_table->add_dt(decision_tab);
        rmse = 0;

        #pragma omp parallel if(par_validation)
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
        }

        rmse /= validation_size;
        rmse = std::sqrt(rmse);
        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_nr_of_dt = tab;
        }
    }

    // std::cout << "Avg. time spent on last part:\t" << time_holder/time_counter << "ms.\n";

    bdt_table->set_optimal_nr_tables(best_nr_of_dt);
    // std::cout << "Optimal nr of dt: " << best_nr_of_dt+1 << "\n";
    // std::cout << "Time dt: " << dt_elapsed.count() << "ms.\n";
    return bdt_table;
}

template<unsigned d, unsigned short t>
double test(matrix &test_set, std::vector<float> &ground_truth, bdt_scoring<d,t> *bdt_table,
        float gt_mean, float gt_std){
    unsigned long test_size = test_set.size();
    double rmse = 0;
    //int correct=0;

    //std::cout << "\n\nTESTING:\n";
    #pragma omp parallel if(par_test)
    {
        #pragma omp for schedule(static) reduction(+:rmse)
        for (int i = 0; i < test_size; ++i) {
            //if(i==0) bdt_table.looper(test_set[i]);
            //if(i==0) std::cout << "PREDICTION: " << ground_truth[0] << ", GT: " << ground_truth[0] << "\n";
            float resp = bdt_table->predict(test_set[i]);
            rmse += std::pow(((resp - ground_truth[i]) - gt_mean) / gt_std, 2);
        }
    }
    rmse /= test_size;
    //std::cout << "Guessed: " << correct << "/" << test_size << " (" << (float)correct/(float)test_size << ")\n";
    return std::sqrt(rmse);
}
#endif //HPC_BDT_BDT_H
