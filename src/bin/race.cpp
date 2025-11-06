#include <matrix.hpp>
#include <vector>
#include <network2.hpp>
#include <functions.hpp>
#include <utility>
#include <stdio.h>
#include <chrono>

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
    using namespace std::chrono;

    Network network = define_network(
        {
            {4, activation_fn::Null},
            {8, activation_fn::ReLU},
            {30, activation_fn::ReLU},
            {16, activation_fn::Sigmoid},
            {16, activation_fn::Sigmoid}
        }, cost_fn::SquaredError, output_type::Dist
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

    auto start = high_resolution_clock::now();
    network.train(5000, training_data, 0.78);
    auto stop = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(stop - start);
    printf("Training done in %lld us\n", dur.count());
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