#pragma once

#include <span>
#include <memory>
#include <vector>
#include <initializer_list>

class Matrix {
    std::vector<double> _data;
    size_t rows;
    size_t cols;
    size_t length;

    public:
    // Matrix(const Matrix& matrix);
    explicit Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, std::initializer_list<double> init);
    Matrix(double val, size_t rows, size_t cols);
    
    std::span<const double> operator[](size_t row) const;
    std::span<double> operator[](size_t row);
    size_t row_size() const;
    size_t col_size() const;
    size_t size() const;

    // Matrix ops
    Matrix& operator+=(const Matrix& rhs);
    friend Matrix operator+(Matrix& lhs, const Matrix& rhs);

    Matrix& operator-=(const Matrix& rhs);
    friend Matrix operator-(Matrix& lhs, const Matrix& rhs);

    Matrix& operator*=(const Matrix& rhs);
    friend Matrix operator*(Matrix& lhs, const Matrix& rhs);

    Matrix& operator*=(double scalar);
    friend Matrix operator*(Matrix& lhs, double scalar);

    Matrix& operator=(const Matrix& other);
};
