
#ifndef HPC_BDT_STRUCTS_H
#define HPC_BDT_STRUCTS_H


#include <cstring>
#include <sys/param.h>


template <unsigned short d>
struct dt{
    // TODO: usare matrice invece di 2 vettori per spazialita cache
    int features[d];
    float cuts[d];
    float predictions[1<<d]; //size 2^d
    float *father_prediction; // TODO: fixed size to avoid many allocations

    dt(){
        memset(features, 0, d*sizeof(unsigned int));
        memset(cuts, 0, d*sizeof(float));
        memset(predictions, 0, (1<<d)*sizeof(float));
        father_prediction = (float*)malloc(sizeof(float));
        memset(father_prediction, 0, sizeof(float));
    }

    void fill_level(int L[], const std::vector<float>& response, int feat, float cut, short level){
        features[level] = feat;
        cuts[level] = cut;
        update_predictions(L, response, level);
    }

    void update_predictions(int L[], const std::vector<float>& response, int level){
        unsigned long N = response.size();
        int count[1<<(level+1)];
        memset(count, 0, (1<<(level+1))*sizeof(int));
        memset(predictions, 0, (1<<d)*sizeof(float));
        if(level < d-1) {
            for (int i = 0; i < N; ++i) {
                predictions[L[i] >> 1] += response[i];
                ++count[L[i] >> 1];
            }
        }else{
            for (int i = 0; i < N; ++i) {
                predictions[L[i]] += response[i];
                ++count[L[i]];
            }
        }
        for(int i=0; i<(1<<(level+1)); ++i) {
            if (count[i] > 0) {
                predictions[i] /= count[i];
            }else{
                // take fathers prediction
                predictions[i] = father_prediction[i>>1];
            }
        }
        delete[] father_prediction;
        father_prediction = (float*)malloc((1<<(level+1))*sizeof(float));
        for(int i=0; i<(1<<(level+1)); ++i)
            father_prediction[i] = predictions[i];
    }


    void printer(){
        for(int i=0; i<d; ++i) std::cout << features[i] << "\t-> " << cuts[i] << std::endl;
        for(int i=(1<<d)-1; i>=0; --i) std::cout << "(" << i << "):" << predictions[i] << " ";
        std::cout << std::endl;
    }

    float predict(const std::vector<float>& feature){
        int idx = 0;
        for(unsigned i=0; i<d; ++i){
            if (feature[features[i]] > cuts[i])
                idx |= (1<<i);
        }
        return predictions[idx];
    }

    float printpredict(const std::vector<float>& feature){
        int idx = 0;
        std::cout << "feature: ";
        for(const float &f: feature)
            std::cout << f << " ";
        std::cout << "\n";
        for(unsigned i=0; i<d; ++i){
            if (feature[features[i]] > cuts[i]) {
                idx |= (1 << i);
                std::cout << "feature " << features[i] << " cut: " << cuts[i] << " orred " << (1<<i) << "\n";
            }
        }
        std::cout << "predicted idx: " << idx << " value: " << predictions[idx] << "\n";
        return predictions[idx];
    }
};


template<unsigned short d>
struct bdt_scoring{
    std::vector<dt<d>> dts;

    void add_dt(dt<d> new_dt){
        dts.push_back(new_dt);
        // new_dt.printer();
    }

    float predict(const std::vector<float>& x){
        float prediction=0;
        for(typename std::vector<dt<d>>::iterator it = dts.begin(); it != dts.end(); ++it){
            prediction += (*it).predict(x);
        }
        return prediction;
    }

    void looper(const std::vector<float>& x){
        for(typename std::vector<dt<d>>::iterator it = dts.begin(); it != dts.end(); ++it){
            (*it).printer();
            (*it).printpredict(x);
            std::cout << "\n";
        }
    }
};


#endif //HPC_BDT_STRUCTS_H
