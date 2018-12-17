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

typedef std::vector<std::vector<float>> matrix;
typedef std::vector<std::vector<int>> imatrix;

/**
 * static
 * const
 * moved things in heap instead of stack, because of memory (problems)
 */

// returns (gain, c) on that feature
// shift=1 when creating decision table, shift>1 when backfitting
std::tuple<long double, int> gain_on_feature(const int nr_leaves, const unsigned long N, const std::vector<int>& L,
                                        std::vector<float>& response,
                                        const std::vector<int>& sorted_feature,
                                        const std::vector<int>& run, const int shift) {
    int act, idx, best_idx=0; //TODO probably shift should be +1
    long double g=0, best_gain=-LDBL_MAX;
    std::vector<int> count(nr_leaves, 0);
    std::vector<double> summer(nr_leaves, 0);

    for (int m = 0; m < N; ++m) {
        count[L[m]] += 1;
        summer[L[m]] += response[m];
    }

    auto doc_id = sorted_feature.begin();

    idx=0;
    // loop on nr of values of that feature
    for (const int &r : run) {
        act=0;
        // update count and summer
        // by changing position of docs having that value on feature j
        while(act!=r){
            count[L[*doc_id]] -= 1;
            summer[L[*doc_id]] -= response[*doc_id];
            count[L[*doc_id]-shift] += 1;
            summer[L[*doc_id]-shift] += response[*doc_id];
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
// Cyclic backfitting
template<unsigned short d>
void backfitting(dt<d>* dt, const matrix& features, const imatrix& sorted_features,
                  const imatrix& runs, std::vector<int>& L, std::vector<float> &response, const int passes){
    unsigned long N = sorted_features[0].size();
    unsigned long n_feat = sorted_features.size();
    long double best_gain=-LDBL_MAX;
    int best_idx=0, best_feature=-1;

    for(int pass=0; pass<passes; ++pass){
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
    }
    (*dt).update_predictions(L, response, d-1);
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
        #pragma omp parallel if(par_dt)
        {
            #pragma omp for schedule(dynamic)
            for(int j=0; j<n_feat; ++j){
                // loop on nr of values of that feature
                // returns a tuple with (gain, idx on sorted feature j)
                std::tuple<long double, int> gc =
                        gain_on_feature(nr_leaves, N, L, response, sorted_features[j], runs[j], 1);
                // TODO critical section or selection of best gain
                if(std::get<0>(gc) > best_gain){
                    best_gain = std::get<0>(gc);
                    best_idx = std::get<1>(gc);
                    best_feature = j;
                }
            } // end loop on feature
        };

        // int mask = 0x00000001;
        auto doc_id = sorted_features[best_feature].begin();
        auto end = sorted_features[best_feature].end();
        auto best_point = doc_id+best_idx;
        if(t < d-1) {
            // update docs which have feature x_j > c
            while(doc_id!=best_point){
                L[*doc_id] -= 1;
                L[*doc_id] <<= 1;
                L[*doc_id] += 1;
                ++doc_id;
            }
            // update docs with feature x_j <= c
            while(doc_id!=end){
                L[*doc_id] <<= 1;
                L[*doc_id] += 1;
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
    //result_dt.printer();
    backfitting(&result_dt, features, sorted_features, runs, L, response, 1);
    return result_dt;
}



template<unsigned d, unsigned short t>
bdt_scoring<d, t>* train(matrix& training_set, matrix& transposed_features, imatrix& sorted_feats,
                     imatrix& runs, std::vector<float>& response){
    unsigned long N = training_set.size();
    bdt_scoring<d, t> *bdt_table;
    bdt_table = new bdt_scoring<d,t>();
    std::vector<float> residuals(N, 0);
    //float rmse=0;

    // TODO remove this transpose, already done in main
    //std::vector<std::vector<float>> transposed_features(n_features, std::vector<float>(N, 0));
    //transpose(training_set, transposed_features);

    // TODO: add criterion to stop nr of tables
    dt<d> initial_dt = create_dt<d>(transposed_features, sorted_feats, runs, response);
    bdt_table->add_dt(initial_dt);
    for(int e=0; e<N; ++e){
        residuals[e] = response[e]-initial_dt.predict(training_set[e]);
        //rmse += std::pow(initial_dt.predict(training_set[e])-response[e], 2);
    }

    //rmse /= N;
    //rmse = std::sqrt(rmse);
    //std::cout << "RMSE at lv 0: " << rmse << std::endl;
    //initial_dt.printer();

    for(int tab=1; tab<t; ++tab){
        dt<d> decision_tab = create_dt<d>(transposed_features, sorted_feats, runs, residuals);
        bdt_table->add_dt(decision_tab);
        //rmse=0;
        for(int e=0; e<N; ++e) {
            float nrm = bdt_table->predict(training_set[e]);
            //rmse += std::pow(nrm-response[e], 2);
            residuals[e] = response[e] - bdt_table->predict(training_set[e]);
            //residuals[e] = response[e] - bdt_table->predict(training_set[e]);
        }
        //rmse /= N;
        //rmse = std::sqrt(rmse);
        //std::cout << "RMSE at lv " << tab << ": " << rmse << std::endl;
    }

    return bdt_table;
}

template<unsigned d, unsigned short t>
double test(matrix &test_set, std::vector<float> &ground_truth, bdt_scoring<d,t> *bdt_table){
    unsigned long test_size = test_set.size();
    double rmse = 0;
    //int correct=0;

    //std::cout << "\n\nTESTING:\n";
    for(int i=0; i<test_size; ++i) {
        //if(i==0) bdt_table.looper(test_set[i]);
        //if(i==0) std::cout << "PREDICTION: " << ground_truth[0] << ", GT: " << ground_truth[0] << "\n";
        float resp = bdt_table->predict(test_set[i]);
        rmse += std::pow(resp-ground_truth[i], 2);
        //if(std::abs(resp-ground_truth[i]) < ground_truth[i]*0.2) ++correct;
        //if(i % 197 == 0) std::cout << "Pred: " << resp
        //<< "\tGt: " << ground_truth[i]
        //<< "\tDiff: " << std::abs(resp-ground_truth[i]) << "\n";
    }
    rmse /= test_size;
    //std::cout << "Guessed: " << correct << "/" << test_size << " (" << (float)correct/(float)test_size << ")\n";
    return std::sqrt(rmse);
}
#endif //HPC_BDT_BDT_H
