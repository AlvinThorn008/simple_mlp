#pragma once

#include <matrix.hpp>
#include <span>
#include <utility>

typedef Matrix (*Activation)(const Matrix&);

enum class activation_fn {
    Null,
    Sigmoid,
    ReLU,
    Softmax,
};

enum class cost_fn {
    SquaredError,
    CrossEntropy
};

enum class output_type {
    Dist,
    General
};

struct LayerParams {
    Matrix weights;
    Matrix bias;

    LayerParams(size_t input_size, size_t output_size);
    LayerParams(Matrix weights, Matrix bias);
};

class Layer {
    public:
    LayerParams params;
    Activation activation;
    Activation diff_activation;

    Layer(size_t input_size, size_t output_size, bool is_random = true);
};

struct LayerDefs {
    unsigned int num_nodes;
    activation_fn activation = activation_fn::Null;
};

class Network {
    std::vector<Matrix> z_values;
    std::vector<Matrix> activations;
    std::vector<Layer> layers;
    std::vector<LayerParams> deltas;
    double (*cost_func)(const Matrix&, const Matrix&);
    Matrix (*dcost_func)(const Matrix&, const Matrix&);
    Matrix (Network::*output_err)(const Matrix&);

    void backward_prop(Matrix target);
    Matrix output_error(const Matrix& target);
    Matrix output_error_softcross(const Matrix& target);

    public:
    Matrix& forward_prop(Matrix input);

    void train(int iters, std::span<std::pair<const Matrix, const Matrix>> batch, double eta);

    friend Network define_network(std::vector<LayerDefs> layers, cost_fn cost_function, output_type output_def);  
};

Network define_network(std::vector<LayerDefs> layers, cost_fn cost_function, output_type output_def); 


class AdamOptimizer {
    double learning_rate, alpha, beta1, beta2;
    std::vector<LayerParams> moment1;
    std::vector<LayerParams> moment2;

    public:
    AdamOptimizer(double learning_rate, double alpha, double beta1, double beta2);
};

