#include <matrix.hpp>
#include <vector>
#include <network.hpp>
#include <tuple>
#include <stdio.h>

Matrix decode(ANN& network, unsigned int input) {
    Matrix mat(0.0, 4, 1);
    for (int j = 3; j >= 0; j--) 
        mat.data()[3 - j] = (double)((input >> j) & 1);

    return network.forward_prop(mat);
}

int main() {
    ANN network{4, 8, 8, 16};

    std::vector<std::tuple<Matrix, Matrix>> training_data;
    training_data.reserve(16);
    for (int i = 0; i < 16; i++) {
        Matrix mat(0.0, 4, 1);
        Matrix out(0.0, 16, 1);
        for (int j = 3; j >= 0; j--) mat.data()[3 - j] = (double)((i >> j) & 1);
        out.data()[i] = 1.0;
        training_data.push_back({mat, out});
        // printf("%d: ", i);
        // print_mat(mat);
    }

    network.sgd(1000, training_data, 0.45);
    for (unsigned int i = 0; i < 16; i++) {
        printf("Input = %u\n", i);
        Matrix output = decode(network, i);
        size_t idx = nn_utils::argmax(output);
        printf("Output = ");
        print_mat(transpose(output));
        printf("Output argmax = %llu\n", idx);
    }

    return 0;
}