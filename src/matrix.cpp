#include "matrix.hpp"
#include <cassert>

Matrix::Matrix(size_t rows, size_t cols) 
    : rows(rows), cols(cols), _data(rows * cols) {}

Matrix::Matrix(double val, size_t rows, size_t cols) 
    : rows(rows), cols(cols), _data(val, rows * cols) {}

Matrix::Matrix(size_t cols, std::initializer_list<double> init)
    : rows(init.size() / cols), cols(cols), _data(init) {}

std::slice_array<double> Matrix::operator[](size_t row) {
    return _data[std::slice(cols * row, cols, 1)];
}

double& Matrix::operator()(size_t r, size_t c) { return _data[c + r * cols]; }

std::slice_array<double> Matrix::column(size_t col) {
    return _data[std::slice(col, cols, cols)];
}

size_t Matrix::row_size() const { return rows; }
size_t Matrix::col_size() const { return cols; }
size_t Matrix::size() const { return _data.size(); }

Matrix& Matrix::operator+=(const Matrix& rhs) {
    rhs.size();
    _data += rhs._data;
    return *this;
}
Matrix operator+(Matrix& lhs, const Matrix& rhs) {
    
}

// Matrix& operator-=(const Matrix& rhs);
// friend Matrix operator-(Matrix& lhs, const Matrix& rhs);

// Matrix& operator*=(const Matrix& rhs);
// friend Matrix operator*(Matrix& lhs, const Matrix& rhs);

// Matrix& operator*=(double scalar);
// friend Matrix operator*(Matrix& lhs, double scalar);

