#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>

#include "structs.h"
#include "utility.h"

std::vector<std::vector<float>> extract_data(std::string filename, std::vector<int> *response, int* N){
    std::vector<std::vector<float>> features;
    std::vector<float> line_f;
    int i=0;
    std::string line;

    // 11 feat vects + 1 response
    std::ifstream file;
    file.open(filename);
    if(file.good())
        std::getline(file, line, '\n'); // header

    if(file.good()) {
        while (std::getline(file, line, ';')) {
            ++i;
            line_f.push_back(std::stof(line));
            if (i == 11) {
                features.push_back(line_f);
                line_f.clear();
                std::getline(file, line, '\n');
                response->push_back((int) std::stof(line));
                i = 0;
                *N = *N + 1;
            }
        }
    }
    file.close();
    return features;
}

void create_dt(std::vector<std::vector<float>> features, std::vector<int> response, unsigned short d){

}

int main(int argc, char* argv[]){
    // init chrono
    std::vector<int> response;
    std::string file = "/home/felix/Desktop/universita/master/high-performance-computing/HPC-BDT/winequality-white.csv";
    int N=0;
    float sum=0;
    std::vector<std::vector<float>> data = extract_data(file, &response, &N);

    float one_d[11*N];
    for(int j=0; j<11; ++j){
        for(int i=0; i<N; ++i){
            one_d[j*N+i] = data[i][j];
        }
    }
    auto begin = std::chrono::high_resolution_clock::now();
    float var_one[N];
    unsigned int i=0, j=0;
    while(i!=N) {
        var_one[i] = one_d[i];
        ++i;
    }

    sequential_sort(var_one, N);

    i=0;
    while(i!=N && j!=N){
        if(var_one[i]!=var_one[j]){
            ++i;
            var_one[i] = var_one[j];
        }
        ++j;
    }
    j=0;
    while(j!=i)
        std::cout << var_one[j++] << std::endl;


    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin);

    std::cout << "Time:\t" << elapsed.count() << "ms." << std::endl;
}
