
#ifndef HPC_BDT_UTILITY_H
#define HPC_BDT_UTILITY_H

//#include "bdt.h" //TODO: why this include + move extract data here

#define swape(x,y){temp=x;x=y;y=temp;}
#define swtemp(x,y,z){temp=x;x=y;y=temp;}

#define parallel_sort true
#define par_dt true
#define par_backfitting true
#define par_validation true
#define par_test false
#define par_dt2 false
#define par_update false

#define nr_backfitting_passes 1

#define chrono_now std::chrono::high_resolution_clock::now()
#define chrono_diff(b,e) std::chrono::duration_cast<std::chrono::milliseconds>(e-b)
#define chrono_prec(b,e) std::chrono::duration_cast<std::chrono::microseconds>(e-b)

int n_threads;
bool enable_par;

void transpose(std::vector<std::vector<float>> &data, std::vector<std::vector<float>> &transposed){
    for(int i=0; i<data.size(); ++i){
        for(int j=0; j<data[0].size(); ++j){
            transposed[j][i] = data[i][j];
        }
    }
}

void shuffle_data(std::vector<std::vector<float>> &data, std::vector<float> &gt){
    unsigned long n = data.size();
    std::vector<float> temp;
    float ftemp;
    while(n != 1){
        int k = static_cast<int>(rand()%n);
        --n;
        std::swap(gt[k],gt[n]);
        swtemp(data[k],data[n],temp);
    }
}

float calc_mean(std::vector<float> &data){
    float mean=0;
    float n = data.size();
    for(int idx=0; idx<n; ++idx){
        mean+= data[idx]/n;
    }
    return mean;
}

float calc_std(std::vector<float> &data, float mean){
    float std=0;
    float n = data.size();
    for(int idx=0; idx<n; ++idx){
        std += (std::pow(data[idx]-mean, 2)/n);
    }
    return std::sqrt(std);
}

// odd-even sort in sequential mode
void sequential_sort(std::vector<float>* data, unsigned int size){
    bool isSorted = false;
    float temp;
    while (!isSorted){
        isSorted = true;

        for (int i=1; i<=size-2; i=i+2){
            if ((*data)[i] < (*data)[i+1]){
                swape((*data)[i],(*data)[i+1])
                isSorted = false;
            }
        }
        for (int i=0; i<=size-2; i=i+2){
            if ((*data)[i] < (*data)[i+1]){
                swape((*data)[i],(*data)[i+1])
                isSorted = false;
            }
        }
    }
}

void odd_even_index_sort(std::vector<float> &data, std::vector<int> &indices){
    bool isSorted = false;
    int size = data.size();

    int temp=0;
    while (!isSorted){
        isSorted = true;
        for (int i=1; i<=size-2; i=i+2){
            if (data[indices[i]] < data[indices[i+1]]){
                //swape(indices[i],indices[i+1]);
                temp = indices[i];
                indices[i] = indices[i+1];
                indices[i+1] = temp;
                isSorted = false;
            }
        }
        for (int i=0; i<=size-2; i=i+2){
            if (data[indices[i]] < data[indices[i+1]]){
                //swape(indices[i],indices[i+1])
                temp = indices[i];
                indices[i] = indices[i+1];
                indices[i+1] = temp;
                isSorted = false;
            }
        }
    }
}


// TODO check bob correctness
void sort_features(std::vector<std::vector<float>>& data,
                   std::vector<std::vector<int>>& runs,
                   std::vector<std::vector<int>>& sorted){
    int n_feats = data[0].size();
    int N = data.size();
    #pragma omp parallel for if(parallel_sort) schedule(dynamic)
        for (int feat=0; feat<N; ++feat) {
            int count=0;
            std::vector<int> v(n_feats);
            std::vector<int> bob;

            std::iota(v.begin(), v.end(), 0);
            //odd_even_index_sort(data[feat], v);
            // TODO: execution policy
            std::sort(v.begin(), v.end(), [&](int i, int j) { return data[feat][i] > data[feat][j]; });
            #pragma omp critical
                sorted[feat].assign(v.begin(), v.end());
            for (int i=0; i<n_feats-1; ++i) {
                if (data[feat][v[i]] == data[feat][v[i+1]])
                    ++count;
                else {
                    bob.push_back(count + 1);
                    count = 0;
                }
            }
            bob.push_back(count+1);
            #pragma omp critical
                runs[feat].assign(bob.begin(), bob.end());
            bob.clear();
        }
}

#endif //HPC_BDT_UTILITY_H
