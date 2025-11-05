#include <matrix.hpp>

namespace nn_funcs {
    Matrix sigmoid(const Matrix& m);
    Matrix dsigmoid(const Matrix& m);

    Matrix softmax(const Matrix& m);
    Matrix dsoftmax(const Matrix& m);

    size_t argmax(const Matrix& m);

    Matrix relu(const Matrix& m);
    Matrix drelu(const Matrix& m);

    double cross_entropy(const Matrix& y1, const Matrix& y2);

    Matrix dcross_entropy(const Matrix& y1, const Matrix& y2);

    double squared_error(const Matrix& y1, const Matrix& y2);
    Matrix dsquared_error(const Matrix& y1, const Matrix& y2);
}