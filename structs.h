
#ifndef HPC_BDT_STRUCTS_H
#define HPC_BDT_STRUCTS_H


#include <cstring>
#include <sys/param.h>
#define shrink 0.05


template <unsigned short d>
struct dt{
    // TODO: usare matrice invece di 2 vettori per spazialita cache
    int features[d];
    float cuts[d];
    float predictions[1<<d]; //size 2^d
    float father_prediction[1<<(d-1)]; // TODO: fixed size to avoid many allocations

    dt(){
        memset(features, 0, d*sizeof(unsigned int));
        memset(cuts, 0, d*sizeof(float));
        memset(predictions, 0, (1<<d)*sizeof(float));
        memset(father_prediction, 0, (1<<(d-1))*sizeof(float));
    }

    ~dt() = default;

    void fill_level(std::vector<int> &L, std::vector<float> &response, int feat, float cut, short level){
        features[level] = feat;
        cuts[level] = cut;
        // TODO why updating if not last level? because of father!
        update_predictions(L, response, level);
    }

    void update_predictions(std::vector<int> &L, std::vector<float> &response, int level){
        unsigned long N = response.size();
        int count[1<<(level+1)];
        memset(count, 0, (1<<(level+1))*sizeof(int));
        memset(predictions, 0, (1<<d)*sizeof(float));
        if(level < d-1) {
            for (int i = 0; i < N; ++i) {
                predictions[L[i] >> 1] += response[i];
                ++count[L[i] >> 1];
            }
            for(int i=0; i<(1<<(level+1)); ++i) father_prediction[i] = predictions[i];
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
    }


    void printer(){
        for(int i=0; i<d; ++i) std::cout << features[i] << "\t-> " << cuts[i] << std::endl;
        for(int i=(1<<d)-1; i>=0; --i) std::cout << "(" << i << "):" << predictions[i] << " ";
        std::cout << std::endl;
    }

    float predict(const std::vector<float>& feature){
        int idx = 0;
        for(unsigned i=0; i<d; ++i){
            if (feature[features[i]] <= cuts[i])
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
            if (feature[features[i]] <= cuts[i]) {
                idx |= (1 << i);
                std::cout << "feature " << features[i] << " cut: " << cuts[i] << " orred " << (1<<i) << "\n";
            }
        }
        std::cout << "predicted idx: " << idx << " value: " << predictions[idx] << "\n";
        return predictions[idx];
    }
};


template<unsigned short d, unsigned short t>
struct bdt_scoring{
    std::vector<dt<d>> dts;
    int pos;

    bdt_scoring(){
        pos=0;
        dts.reserve(t);
    }

    ~bdt_scoring() = default;

    void add_dt(dt<d> new_dt){
        dts.push_back(new_dt);
        ++pos;
        //dts.push_back(*new_dt);
        // new_dt.printer();
    }

    float predict(const std::vector<float>& x){
        auto beg = dts.begin();
        auto end = dts.end();
        //float prediction=(*beg).predict(x);
        //if(beg==end) return prediction;
        float prediction=0;

        for(typename std::vector<dt<d>>::iterator it = beg; it != end; ++it){
            prediction += shrink*(*it).predict(x);
        }
        return prediction;
    }

    void looper(const std::vector<float>& x){
        for(typename std::vector<dt<d>>::iterator it = (*dts).begin(); it != (*dts).end(); ++it){
            (*it).printer();
            (*it).printpredict(x);
            std::cout << "\n";
        }
    }
};


#endif //HPC_BDT_STRUCTS_H
