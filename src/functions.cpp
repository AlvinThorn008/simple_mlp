#include <functions.hpp>
#include <cmath>
#include <cassert>

namespace nn_funcs {
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

    Matrix softmax(const Matrix& m) {
        assert(m.col_count() == 1 && "softmax expects a column vector");
        Matrix res(m.row_count(), m.col_count());
        double sum = 0.0;
        double max_val = m.data()[argmax(m)];
        for (size_t i = 0; i < m.data().size(); i++) 
            sum += exp(m.data()[i] - max_val);

        for (size_t i = 0; i < m.data().size(); i++) 
            res.data()[i] = exp(m.data()[i] - max_val) / sum;

        return res;
    }
    Matrix dsoftmax(const Matrix& m) { return m; }

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

    Matrix relu(const Matrix& m) {
        Matrix res(0.0, m.row_count(), m.col_count());
        for (size_t i = 0; i < m.size(); i++) res.data()[i] = __max(m.data()[i], 0.0);
        return res;
    }

    Matrix drelu(const Matrix& m) {
        Matrix res(0.0, m.row_count(), m.col_count());
        for (size_t i = 0; i < m.size(); i++) res.data()[i] = (double)(m.data()[i] > 0.0);
        return res;
    }

    double cross_entropy(const Matrix& y1, const Matrix& y2) {
        assert(y1.row_count() == y2.row_count() && (y1.col_count() == 1) && "cross_entropy expects a column vector");
        
        double sum = 0.0;
        for (size_t i = 0; i < y1.row_count(); i++) sum -= y2.data()[i] * log(y1.data()[i]);
    
        return sum;
    }

    Matrix dcross_entropy(const Matrix& y1, const Matrix& y2) {
        assert(y1.row_count() == y2.row_count() && (y1.col_count() == 1) && "dcross_entropy expects a column vector");

        Matrix res(0.0, y1.row_count(), y1.col_count());
        for (size_t i = 0; i < y1.row_count(); i++) res.data()[i] = -y2.data()[i] / (y1.data()[i] + 1e-9);
        return res;
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
}