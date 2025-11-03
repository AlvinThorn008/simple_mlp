#pragma once

#include <span>
#include <memory>
#include <vector>
#include <initializer_list>

class Matrix {
    // Constructor initializer lists are initialized in the order defined
    // in the class 
    // e.g. Matrix(...) : length(3), _data(length) would pass an uninitialized `length`
    // were `std::vector<double> _data` to be declared before `size_t length`
    size_t rows, cols, length;
    std::vector<double> _data;

    public:
    explicit Matrix(size_t rows, size_t cols);
    Matrix(size_t cols, std::initializer_list<double> init);
    Matrix(double val, size_t rows, size_t cols);
    
    std::span<const double> operator[](size_t row) const;
    std::span<double> operator[](size_t row);
    std::span<const double> data() const;
    std::span<double> data();
    size_t row_count() const;
    size_t col_count() const;
    size_t size() const;

    // Matrix ops

    friend Matrix transpose(Matrix mat);

    friend Matrix hadamard(Matrix lhs, const Matrix& rhs);

    Matrix& operator+=(const Matrix& rhs);
    friend Matrix operator+(Matrix lhs, const Matrix& rhs);

    Matrix& operator-=(const Matrix& rhs);
    friend Matrix operator-(Matrix lhs, const Matrix& rhs);

    Matrix& operator*=(const Matrix& rhs);
    friend Matrix operator*(Matrix lhs, const Matrix& rhs);

    Matrix& operator*=(double scalar);
    friend Matrix operator*(Matrix lhs, double scalar);

    // Specialized matrix ops
    Matrix& add_col(const Matrix& rhs);

};

void print_mat(const Matrix& mat);
