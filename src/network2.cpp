#include <network2.hpp>
#include <matrix.hpp>
#include <random>
#include <cassert>
#include <utility>
#include <functions.hpp>

// #define NN_DIAG


LayerParams::LayerParams(size_t input_size, size_t output_size) 
: weights(0.0, output_size, input_size), bias(0.0, output_size, 1) {}

LayerParams::LayerParams(Matrix weights, Matrix bias) : weights(weights), bias(bias) {}


Layer::Layer(size_t input_size, size_t output_size, bool is_random): 
params(input_size, output_size) {

    if (!is_random) return;
    std::random_device rd{}; // Get initial seed from hardware random source
    std::mt19937 gen{rd()}; // Mersenne Twister
    std::normal_distribution d1{0.0, sqrt(2.0 / (input_size + output_size))};
    std::normal_distribution d2{0.0, sqrt(2.0 / (input_size + output_size))};
    
    for (size_t i = 0; i < params.weights.size(); i++) params.weights.data()[i] = d1(gen);
    for (size_t i = 0; i < params.bias.size(); i++) params.bias.data()[i] = d2(gen);
}

void Network::backward_prop(Matrix target) {
    int l = layers.size() - 1;

    Matrix grad = (*this.*output_err)(target);
    deltas[l].bias = grad;
    
    deltas[l].weights = mmrt(grad, activations[l]);

    while ((--l) >= 0) {
        grad = hadamard(layers[l].diff_activation(z_values[l+1]), transpose(layers[l+1].params.weights) * grad); 
        deltas[l].bias = grad;
        deltas[l].weights = mmrt(grad, activations[l]);
    }
}

Matrix Network::output_error(const Matrix& target) {
    int l = layers.size() - 1;
    return hadamard((layers[l].diff_activation)(z_values[l+1]), (dcost_func)(activations[l+1], target));
}

Matrix Network::output_error_softcross(const Matrix& target) {
    int l = activations.size() - 1;

    return activations[l] - target;
}

Matrix& Network::forward_prop(Matrix input) {
    z_values[0] = activations[0] = input;
    for (size_t l = 1; l < z_values.size(); ++l) {
        z_values[l] = layers[l - 1].params.weights * activations[l - 1]
            + layers[l - 1].params.bias;
        activations[l] = layers[l - 1].activation(z_values[l]);
    }
    return activations.back();
}

void Network::train(int iters, std::span<std::pair<const Matrix, const Matrix>> batch, double eta) {
    for (int iter = 0; iter < iters; iter++) {
        std::vector<LayerParams> delta_sum;
        delta_sum.reserve(layers.size());
        
        // Zero out delta_sum in layers' structure
        for (size_t i = 0; i < layers.size(); i++) {
            delta_sum.push_back({
                Matrix(layers[i].params.weights.row_count(), layers[i].params.weights.col_count()),
                Matrix(layers[i].params.bias.row_count(), layers[i].params.bias.col_count())
            });
        }
        // Backpropagation
        for (const auto& example : batch) {
            forward_prop(example.first);
            backward_prop(example.second);
            for (size_t i = 0; i < layers.size(); i++) {
                delta_sum[i].bias += deltas[i].bias;
                delta_sum[i].weights += deltas[i].weights;
            }
        }
        // Update
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].params.bias -= delta_sum[i].bias * (eta / (double)batch.size());
            layers[i].params.weights -= delta_sum[i].weights * (eta / (double)batch.size());
        }
        #ifdef NN_DIAG
        printf("Iter %d cost = %lf\n", iter, cost_func(activations.back(), std::get<1>(batch.back())));
        #endif
    }
}

Network define_network(std::vector<LayerDefs> layers, cost_fn cost_function, output_type output_def) {
    assert(layers.size() >= 2 && "define_network: at least 2 layers must be specified.");
    Network network;
    network.layers.reserve(layers.size() - 1);
    network.z_values.reserve(layers.size());
    network.activations.reserve(layers.size());
    network.deltas.reserve(layers.size() - 1);
    
    switch (cost_function) {
        case cost_fn::SquaredError: 
            network.cost_func = nn_funcs::squared_error; 
            network.dcost_func = nn_funcs::dsquared_error; 
            break;
        case cost_fn::CrossEntropy: 
            network.cost_func = nn_funcs::cross_entropy; 
            network.dcost_func = nn_funcs::dcross_entropy; 
            break;
        default: break;
    }

    network.output_err = &Network::output_error;

    for (size_t i = 0; i < layers.size() - 1; i++) {
        std::array<Activation, 2> acts = { nullptr, nullptr };
        switch (layers[i + 1].activation) {
            case activation_fn::Null: break;
            case activation_fn::ReLU: acts = { nn_funcs::relu, nn_funcs::drelu }; break;
            case activation_fn::Sigmoid : acts = { nn_funcs::sigmoid, nn_funcs::dsigmoid }; break;
            case activation_fn::Softmax : { 
                assert(((i + 1 == layers.size() - 1) 
                && (cost_function == cost_fn::CrossEntropy))
                && (output_def == output_type::Dist) && "define_network: softmax currently only supports in output layer with cross entropy and Distribution output");
                network.output_err = &Network::output_error_softcross;
                acts = { nn_funcs::softmax, nn_funcs::dsoftmax };
                break;
            }
            default: break;
        }
        network.deltas.push_back(LayerParams(layers[i].num_nodes, layers[i+1].num_nodes));
        network.layers.push_back(Layer(layers[i].num_nodes, layers[i+1].num_nodes));
        network.activations.push_back(Matrix(layers[i+1].num_nodes, 1));
        network.z_values.push_back(Matrix(layers[i].num_nodes, 1));
        network.layers.back().activation = acts[0];
        network.layers.back().diff_activation = acts[1];
    }
    network.activations.push_back(Matrix(layers.back().num_nodes, 1));
    network.z_values.push_back(Matrix(layers.back().num_nodes, 1));

    return network;
}