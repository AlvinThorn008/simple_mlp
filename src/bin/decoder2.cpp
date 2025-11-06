#include <matrix.hpp>
#include <vector>
#include <network2.hpp>
#include <functions.hpp>
#include <utility>
#include <stdio.h>

Matrix decode(Network& network, unsigned int input) {
    Matrix mat(0.0, 4, 1);
    for (int j = 3; j >= 0; j--) 
        mat.data()[3 - j] = (double)((input >> j) & 1);

    return network.forward_prop(mat);
}

void print_mat2(const Matrix& mat) {
    if (mat.size() == 0) { printf("[]\n"); return; }
    putchar('[');
    for (size_t r = 0; r < mat.row_count(); r++) {
        auto row = mat[r];
        auto tail = row.subspan(1);
        printf(" %-3.2lf", row[0]);
        for (double val: tail) {
            printf(", %-3.2lf", val);
        }
    }
    printf(" ]\n");
}

int main() {
    Network network = define_network(
        {
            {4, activation_fn::Null},
            {8, activation_fn::ReLU},
            {16, activation_fn::Softmax}
        }, cost_fn::CrossEntropy, output_type::Dist
    );

    /* Generate training data */
    std::vector<std::pair<const Matrix, const Matrix>> training_data;
    training_data.reserve(16);
    for (int i = 0; i < 16; i++) {
        Matrix mat(0.0, 4, 1); // Input
        Matrix out(0.0, 16, 1); // Output
        // int to 4-bit binary double vector e.g. 3 -> [0.0, 0.0, 1.0, 1.0]
        for (int j = 3; j >= 0; j--) mat.data()[3 - j] = (double)((i >> j) & 1);
        out.data()[i] = 1.0;
        training_data.push_back({mat, out});
    }

    network.train(50, training_data, 3.35);
    for (unsigned int i = 0; i < 16; i++) {
        printf("Input = %u\n", i);
        Matrix output = decode(network, i); // Decode network output for i
        size_t idx = nn_funcs::argmax(output);
        printf("Output = ");
        print_mat2(transpose(output));
        printf("Output argmax = %llu\n", idx);
    }

    return 0;
}