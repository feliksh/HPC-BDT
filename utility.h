
#ifndef HPC_BDT_UTILITY_H
#define HPC_BDT_UTILITY_H

#define swape(x,y){temp=x;x=y;y=temp;}
#define swtemp(x,y,z){temp=x;x=y;y=temp;}

#define parallel_sort true
#define par_dt 1
#define par_backfitting 2
#define par_validation 3
#define par_test 4
#define par_update 5
#define par_dt2 6

#define nr_backfitting_passes 1
#define history_length 10

#define chrono_now std::chrono::high_resolution_clock::now()
#define chrono_diff(b,e) std::chrono::duration_cast<std::chrono::milliseconds>(e-b)
#define chrono_prec(b,e) std::chrono::duration_cast<std::chrono::microseconds>(e-b)

int par_value;


/**
 *
 * @param data pointer to vector of vectors
 * @param result pointer to memory where datas are going to end up
 * @param N nr of data
 * @param n_feat nr of features
 */
void transform(std::vector<std::vector<float>>& data, float* result, int N, int n_feat){
    for(int i=0; i<N; ++i)
        for(int j=0; j<n_feat; ++j)
            result[(j*N)+i] = data[i][j];
}

// the response value should be the last value of the column
std::vector<std::vector<float>> extract_data(const std::string *filename, std::vector<float> *response,
                                             int* N, unsigned long* n_features, char sep){
    std::vector<std::vector<float>> features;
    std::vector<float> line_f;
    int i=-1;
    *n_features=0;
    std::string line;

    std::ifstream file;
    file.open(*filename);

    // read Header
    if(file.good()) {
        while (i != 0) {
            std::getline(file, line, '\n');
            for (char &it : line) if (it == sep) ++(*n_features);
            i = 0;
        }
    }

    if(file.good()) {
        while (std::getline(file, line, sep)) {
            ++i;
            if(i!=*n_features) line_f.push_back(std::stof(line));
            else{
                line_f.push_back(std::stof(line));
                features.push_back(line_f);
                line_f.clear();
                std::getline(file, line, '\n');
                response->push_back(std::stof(line));
                i = 0;
                *N = *N + 1;
            }
        }
    }
    file.close();

    return features;
}


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
            // TODO: try execution policies for std::sort
            std::sort(v.begin(), v.end(), [&](int i, int j) { return data[feat][i] > data[feat][j]; });
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
            runs[feat].assign(bob.begin(), bob.end());
            bob.clear();
        }
}

#endif //HPC_BDT_UTILITY_H
