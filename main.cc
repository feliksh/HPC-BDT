#include <iostream>
#include <chrono>

int main(int argc, char* argv[]){
    // init chrono
    auto begin = std::chrono::high_resolution_clock::now();

    std::cout << std::endl;
    std::cout << "Life is very Beautiful" << std::endl;
    std::cout << "Hiwot Betam des Tilalech" << std::endl;

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin);

    std::cout << "Time:\t" << elapsed.count() << "ms." << std::endl;
}
