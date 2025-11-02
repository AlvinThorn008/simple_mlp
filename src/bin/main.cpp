#include <matrix.hpp>
#include <stdio.h>

int main() {
    Matrix a(3, {
        2.0,  3.0,  4.0, 
        -1.0, 0.3, -1.2,
        5.0,  -2.3, 2.2,
    }); // 3x3
    Matrix b(3, {
        1, 2, 1,
        2, 0, 1,
        0, 1, 0
    });
    Matrix c(2, {
        0,    2,
        1.5,  0.5,
        2,    1 
    });

    print_mat(a);
    auto mul = b * a;
    print_mat(mul);
    auto mul2 = b * c;
    print_mat(mul);
    a += b;
    print_mat(a);

    return 0;
}