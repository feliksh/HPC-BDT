//
// Created by felix on 21/05/19.
//

#ifndef HPC_BDT_BACKFITTING_H
#define HPC_BDT_BACKFITTING_H

#include "utility.h"
#include "structs.h"

std::tuple<long double, int> gain_on_feature(const int nr_leaves, const unsigned long N, const std::vector<int>& L,
                                             std::vector<float>& response,
                                             const std::vector<int>& sorted_feature,
                                             const std::vector<int>& run, const int shift);

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
        #pragma omp parallel if(par_backfitting)
        {
            #pragma omp for schedule(dynamic)
            for(int j=0; j<n_feat; ++j){
                // loop on nr of values of that feature
                // returns a tuple with (gain, idx on sorted feature j)
                std::tuple<long double, int> gc = gain_on_feature(1<<d, N, L, response, sorted_features[j], runs[j], 1<<(d-t-1));
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


template<unsigned short d>
void backfitting_random(dt<d>* dt, const matrix& features, const imatrix& sorted_features,
                        const imatrix& runs, std::vector<int>& L, std::vector<float> &response){
    unsigned long N = sorted_features[0].size();
    unsigned long n_feat = sorted_features.size();
    long double best_gain=-LDBL_MAX;
    int best_idx=0, best_feature=-1;
    int t=-1, last_t=-1;

    for(int e=0; e<d; ++e){
        best_gain=-LDBL_MAX;
        // choose a random cut and remove it (take care of not choosing last cut->would result in no improvement)
        do {
            t = rand() % d;
        }while(t==last_t);
        last_t = t;
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


template<unsigned short d>
void backfitting_greedy(dt<d>* dt, const matrix& features, const imatrix& sorted_features,
                        const imatrix& runs, std::vector<int>& L, std::vector<float> &response){
    unsigned long N = sorted_features[0].size();
    unsigned long n_feat = sorted_features.size();
    long double best_gain=-LDBL_MAX;
    int best_idx=0, best_feature=-1;
    int best_t=-1;

    for(int e=0; e<d; ++e){
        best_gain=-LDBL_MAX;
        // choose the cut to be removed as the one which gives the best imrpovement
        for(int t=0; t<d; ++t){
            std::vector<int> temp_L(L);
            // remove the cut
            for(int m=0; m<N; ++m) temp_L[m] |= (1<<(d-t-1));
            // loop on features
            #pragma omp parallel if(par_backfitting) shared(N,temp_L,response,sorted_features,runs,t)
            {
                #pragma omp for schedule(dynamic)
                for(int j=0; j<n_feat; ++j){
                    // loop on nr of values of that feature
                    // returns a tuple with (gain, idx on sorted feature j)
                    std::tuple<long double, int> gc =
                            gain_on_feature(1<<d, N, temp_L, response, sorted_features[j], runs[j], 1<<(d-t-1));
                    // TODO: critical section?
                    if(std::get<0>(gc) > best_gain){
                        best_gain = std::get<0>(gc);
                        best_idx = std::get<1>(gc);
                        best_feature = j;
                        best_t = t;
                    }
                } // end loop on feature
            };
        }
        // the best cut is at level 'best_t'

        // remove the cut al level 'best_t'
        for(int m=0; m<N; ++m) L[m] |= (1<<(d-best_t-1));

        auto doc_id = sorted_features[best_feature].begin();
        //auto end = sorted_features[best_feature].end();
        auto best_point = doc_id+best_idx;
        // update docs which have feature x_j > c
        while(doc_id!=best_point){
            L[*doc_id] -= (1<<(d-best_t-1));
            ++doc_id;
        }
        float best_c = features[best_feature][*best_point];
        (*dt).features[best_t] = best_feature;
        (*dt).cuts[best_t] = best_c;
    }
    (*dt).update_predictions(L, response, d-1);
}


#endif //HPC_BDT_BACKFITTING_H
