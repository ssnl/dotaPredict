#include "doublefann.h"
#include <fann.h>
#include "./fann_cpp.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(int argc, char* argv[]) {

    int NUM_FEATURES = 108 * 2;

    string num;

    fann_type input[NUM_FEATURES];

    ifstream file (argv[1]);

    int i = 0;

    while (file.good()) {
        getline(file, num, ',');
        input[i] = ::atof(num.c_str());
        // printf("%d\n", input[i]);
        i++;
    }

    struct fann *ann = fann_create_from_file("./data/train.net");

    fann_type *calc_out = fann_run(ann, input);
    printf("%f", calc_out[0]);
    fann_destroy(ann);
    return 0;
}