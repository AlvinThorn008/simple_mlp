#include "matrix.hpp"
#include <cassert>
#include <algorithm>
#include <stdio.h>

Matrix::Matrix(size_t rows, size_t cols) 
    : rows(rows), cols(cols), length(rows * cols), 
    _data(length) {}

Matrix::Matrix(double val, size_t rows, size_t cols) 
    : rows(rows), cols(cols), length(rows * cols), 
    _data(length, val) {}

Matrix::Matrix(size_t cols, std::initializer_list<double> init)
    : rows(init.size() / cols), cols(cols), length(init.size()),
    _data(init) {
    assert(init.size() % cols == 0 && "initializer list size must be multiple of cols");
}

std::span<double> Matrix::operator[](size_t row) {
    return std::span(_data.begin() + cols * row, cols);
}

std::span<const double> Matrix::operator[](size_t row) const {
    return std::span(_data.begin() + cols * row, cols);
}

size_t Matrix::row_size() const { return rows; }
size_t Matrix::col_size() const { return cols; }
size_t Matrix::size() const { return length; }

Matrix& Matrix::operator+=(const Matrix& rhs) {
    assert(rows == rhs.rows && cols == rhs.cols);
    for (size_t i = 0; i < length; i++) _data[i] += rhs._data[i];
    return *this;
}
Matrix operator+(Matrix& lhs, const Matrix& rhs) {
    lhs += rhs;
    return Matrix(lhs);
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
    assert(rows == rhs.rows && cols == rhs.cols);
    for (size_t i = 0; i < length; i++) _data[i] -= rhs._data[i];
    return *this;
}
Matrix operator-(Matrix& lhs, const Matrix& rhs) {
    lhs -= rhs;
    return Matrix(lhs);
}

Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < length; i++) _data[i] *= scalar;
    return *this;
}
Matrix operator*(Matrix& lhs, double scalar) {
    lhs *= scalar;
    return Matrix(lhs);
}

Matrix& Matrix::operator*=(const Matrix& rhs) {
    *this = (*this * rhs);
    return *this;
}

Matrix operator*(Matrix& lhs, const Matrix& rhs) {
    assert(lhs.cols == rhs.rows);
    Matrix result(lhs.rows, rhs.cols);

    for (size_t row = 0; row < lhs.rows; row++) {
        auto result_row = result[row];
        for (size_t col = 0; col < rhs.cols; col++) {
            double sum = 0;
            auto left_row = lhs[row];
            for (size_t k = 0; k < lhs.cols; k++) {
                sum += left_row[k] * rhs[k][col];
            }
            result_row[col] = sum;
        }
    }
    return result;
}

void print_mat(const Matrix& mat) {
    if (mat.size() == 0) { printf("[]\n"); return; }
    printf("[\n");
    for (size_t r = 0; r < mat.row_size(); r++) {
        auto row = mat[r];
        auto tail = row.subspan(1);
        printf("  %-6.2lf", row[0]);
        for (double val: tail) {
            printf("  %-6.2lf", val);
        }
        printf("\n");
    }
    printf("]\n");
}