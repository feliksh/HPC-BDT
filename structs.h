
#ifndef HPC_BDT_STRUCTS_H
#define HPC_BDT_STRUCTS_H


#include <cstring>
#include <sys/param.h>

/**
template<typename T, unsigned int w, unsigned int h>
class Matrix {
protected:
    T a[w][h];

public:
    Matrix(){
        memset(a, T(0), sizeof(a[0][0])*w*h);
    }

    int width(){ return w; }
    int height(){ return h; }

    // number of occupied bytes of memory
    int memory(){ return sizeof(T)*w*h;}

    T & operator()(unsigned int i, unsigned int j){
        if(i >= w || j >= h){ }

        return a[i][j];
    };
};**/

class Matrix{
protected:
    std::vector<std::vector<float>> a;
    unsigned int width;
    unsigned int height;

public:
    Matrix(){
        width=0; height=0;
    }

    Matrix(unsigned width, unsigned height){
        this->width = width;
        this->height = height;
    }

    Matrix(const Matrix& X){
        this->width = X.width;
        this->height = X.height;
        a = X.a;
    }

    int get_width(){ return width; }
    int get_height(){ return height; }

    float & operator()(unsigned int i, unsigned int j){
        if(i >= width || j >= height){ }

        return a[i][j];
    }

};

template <unsigned short d>
struct dt{
    // TODO: usare matrice invece di 2 vettori per spazialita cache
    unsigned int features[d];
    float cuts[d];
    int predictions[1<<d]; //size 2^d

    dt(){
        memset(features, 0, d*sizeof(unsigned int));
        memset(cuts, 0, d*sizeof(float));
        memset(predictions, 0, (1<<d)*sizeof(float));
    }

    void fill_level(unsigned int feat, float cut, short level){
        features[level] = feat;
        cuts[level] = cut;
    }

    void update_predictions(short L[], std::vector<int>& response, int N){
        int count[1<<d];
        memset(count, 0, (1<<d)*sizeof(int));
        for(int i=0; i<N; ++i){
            predictions[L[i]] += response[i];
            ++count[L[i]];
        }
        for(int i=0; i<(1<<d); ++i) {
            if (count[i] > 0)
                predictions[i] /= count[i];
            // std::cout << "lv " << i << ": " << count[i] << std::endl;
        }
        std::cout << std::endl;
    }

    void printer(){
        for(int i=0; i<d; ++i) std::cout << features[i] << "\t-> " << cuts[i] << std::endl;
        for(int i=0; i<(1<<d); ++i) std::cout << predictions[i] << " ";
        std::cout << std::endl;
    }

    float predict(float* x, int id, int N){
        unsigned int idx = 0;
        for(unsigned i=0; i<d; ++i){
            if (x[(features[i]*N)+id] <= cuts[i])
                idx = (idx << 1) | 1;
        }
        return predictions[idx];
    }
};

#endif //HPC_BDT_STRUCTS_H
