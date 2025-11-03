#pragma once

#include <matrix.hpp>
#include <vector>
#include <initializer_list>

class Layer {
    public:
	Matrix weights;
	Matrix bias;
	Layer(size_t input_size, size_t output_size);
    Layer(Matrix weights, Matrix bias);
};

class ANN {
    std::vector<Layer> layers;
    std::vector<Matrix> activations; // Per layer
    std::vector<Matrix> nodes;
    std::vector<Layer> deltas;

    public:
    // Creates ANN with layer of the sizes list. First element should be the size of the input
    ANN(std::initializer_list<size_t> sizes); 
    Matrix forward_prop(Matrix input);
    void backward_prop(Matrix target);
    void sgd(int iters, std::vector<std::tuple<Matrix, Matrix>> batch, double eta);
};

namespace nn_utils {
    Matrix sigmoid(const Matrix& m);
    Matrix dsigmoid(const Matrix& m);

    Matrix softmax(const Matrix& m);
    Matrix dsoftmax(const Matrix& m);

    size_t argmax(const Matrix& m);

    Matrix relu(const Matrix& m);

    double cross_entropy(const Matrix& y1, const Matrix& y2);
    double cross_entropy_idx(const Matrix& y1, size_t idx);

    Matrix dcross_entropy(const Matrix& y1, size_t idx);

    double squared_error(const Matrix& y1, const Matrix& y2);
    Matrix dsquared_error(const Matrix& y1, const Matrix& y2);
}

/*
X W1 W2 W3
*/