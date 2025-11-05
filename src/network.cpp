#include <network.hpp>
#include <matrix.hpp>
#include <cmath>
#include <cassert>
#include <tuple>
#include <random>


Layer::Layer(size_t input_size, size_t output_size):
    weights(0.0, output_size, input_size), bias(0.0, output_size, 1) {

    std::random_device rd{}; // Get initial seed from hardware random source
    std::mt19937 gen{rd()}; // Mersenne Twister
    std::normal_distribution d1{0.0, sqrt(2.0 / (input_size + output_size))};
    std::normal_distribution d2{0.0, sqrt(2.0 / (input_size + output_size))}; // mean = 5.0, stddev = 2.0
    
    for (size_t i = 0; i < weights.size(); i++) weights.data()[i] = d1(gen);
    for (size_t i = 0; i < bias.size(); i++) bias.data()[i] = d2(gen);
}

Layer::Layer(Matrix weights, Matrix bias):
    weights(weights), bias(bias) {}

ANN::ANN(std::initializer_list<size_t> sizes) {
    if (sizes.size() < 2) return;
    layers.reserve(sizes.size() - 1);
    activations.reserve(sizes.size());
    nodes.reserve(sizes.size());
    deltas.reserve(sizes.size() - 1);
    for (auto size_ptr = sizes.begin(); size_ptr < sizes.end() - 1; size_ptr++) {
        layers.push_back(Layer(*size_ptr, *(size_ptr + 1)));
        deltas.push_back(Layer(*size_ptr, *(size_ptr + 1)));
        activations.push_back(Matrix(*(size_ptr + 1), 1));
        nodes.push_back(Matrix(*(size_ptr), 1));
    }
    activations.push_back(Matrix(*(sizes.end() - 1), 1));
    nodes.push_back(Matrix(*(sizes.end() - 1), 1));
}

Matrix ANN::forward_prop(Matrix input) {
    nodes[0] = input;
    activations[0] = input;
    for (size_t l = 1; l < nodes.size(); l++) {
        nodes[l] = (layers[l - 1].weights * activations[l - 1]) + layers[l - 1].bias;
        activations[l] = nn_utils::sigmoid(nodes[l]);
    }
    return activations.back();
}

void ANN::backward_prop(Matrix target) {
    int l = layers.size() - 1;
    
    Matrix grad = hadamard(nn_utils::dsigmoid(nodes[l+1]), nn_utils::dsquared_error(activations[l+1], target));
    deltas[l].bias = grad;
    
    deltas[l].weights = grad * transpose(activations[l]);
   
    while ((--l) >= 0) {
        grad = hadamard(nn_utils::dsigmoid(nodes[l+1]), transpose(layers[l+1].weights) * grad); 
        deltas[l].bias = grad;
        deltas[l].weights = grad * transpose(activations[l]);
    }
}

void ANN::sgd(int iters, std::vector<std::tuple<Matrix, Matrix>> batch, double eta) {
    for (int iter = 0; iter < iters; iter++) {
        std::vector<Layer> delta_sum;
        delta_sum.reserve(layers.size());
        
        // Zero out delta_sum in layers' structure
        for (size_t i = 0; i < layers.size(); i++) {
            delta_sum.push_back({
                Matrix(layers[i].weights.row_count(), layers[i].weights.col_count()),
                Matrix(layers[i].bias.row_count(), layers[i].bias.col_count())
            });
            // delta_sum[i].bias = ;
            // delta_sum[i].weights = ;
        }
        // Backpropagation
        for (auto example : batch) {
            forward_prop(std::get<0>(example));
            backward_prop(std::get<1>(example));
            for (size_t i = 0; i < layers.size(); i++) {
                delta_sum[i].bias += deltas[i].bias;
                delta_sum[i].weights += deltas[i].weights;
            }
        }
        // Update
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].bias -= delta_sum[i].bias * (eta / (double)batch.size());
            layers[i].weights -= delta_sum[i].weights * (eta / (double)batch.size());
        }
        printf("Iter %d squared_error = %lf\n", iter, nn_utils::squared_error(activations.back(), std::get<1>(batch.back())));
    }
}

namespace nn_utils {
    Matrix sigmoid(const Matrix& m) {
        Matrix res(m.row_count(), m.col_count());
        for (size_t i = 0; i < m.size(); i++) 
            res.data()[i] = 1.0 / (1.0 + exp(-m.data()[i]));
        
        return res;
    }

    Matrix dsigmoid(const Matrix& m) {
        Matrix res(m.row_count(), m.col_count());
        double sig;
        for (size_t i = 0; i < m.size(); i++) {
            sig = 1.0 / (1.0 + exp(-m.data()[i]));
            res.data()[i] = sig * (1 - sig);
        }
        return res;
    }

    size_t argmax(const Matrix& m) {
        assert(m.col_count() == 1 && "argmax expects a column vector");
        double max = -INFINITY;
        size_t max_idx = 0;
        for (size_t i = 0; i < m.size(); i++) {
            if (m.data()[i] >= max) { 
                max = m.data()[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    double cross_entropy(const Matrix& y1, const Matrix& y2) {
        assert(y1.row_count() == y2.row_count() && (y1.col_count() == 1) && "cross_entropy expects a column vector");
        
        double sum = 0.0;
        for (size_t i = 0; i < y1.row_count(); i++) sum -= y2.data()[i] * log(y1.data()[i]);
    
        return sum;
    }

    double squared_error(const Matrix& y1, const Matrix& y2) {
        Matrix res = y1 - y2;
        double sum = 0.0;
        for (double err : res.data()) sum += err * err;
        return sum;
    }

    Matrix dsquared_error(const Matrix& y1, const Matrix& y2) {
        return (y1 - y2) * 2.0;
    }

    double cross_entropy_idx(const Matrix& y1, size_t idx) {
        assert((idx < y1.row_count()) && (y1.col_count() == 1) && "cross_entropy_idx expects a column vector");
        return -log(y1.data()[idx] + 1e-9);
    }

    Matrix dcross_entropy(const Matrix& y1, size_t idx) {
        assert((idx < y1.row_count()) && (y1.col_count() == 1) && "dx cross_entropy expects a column vector");

        Matrix res(0.0, y1.row_count(), y1.col_count());
        res[idx][0] = -1.0 / (y1[idx][0] + 1e-9);
        return res;
    }
}

