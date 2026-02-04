#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/**
 * Gestion de Matrices avec OpenBLAS
 * =================================
 * 
 * Wrapper pour les opérations matricielles utilisant OpenBLAS.
 * Objectif: Maximiser l'utilisation des routines BLAS optimisées.
 */

// Structure de matrice (row-major)
typedef struct {
    double* data;       // Données en ligne (row-major pour compatibilité NumPy)
    int rows;
    int cols;
} Matrix;

// ====================
// Allocation/Libération
// ====================

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* mat);
Matrix* matrix_zeros(int rows, int cols);
Matrix* matrix_ones(int rows, int cols);
Matrix* matrix_random_normal(int rows, int cols, double mean, double std);

// ====================
// Opérations BLAS Level 3 (GEMM - Le plus important!)
// ====================

/**
 * Multiplication matricielle: C = alpha*A*B + beta*C
 * Utilise cblas_dgemm (performance maximale)
 */
void matrix_multiply(const Matrix* A, const Matrix* B, Matrix* C,
                    double alpha, double beta, 
                    int transpose_A, int transpose_B);

// ====================
// Opérations BLAS Level 1
// ====================

/**
 * Addition scalaire: X = X + alpha*Y
 * Utilise cblas_daxpy
 */
void matrix_axpy(double alpha, const Matrix* X, Matrix* Y);

/**
 * Multiplication scalaire: X = alpha*X
 * Utilise cblas_dscal
 */
void matrix_scale(double alpha, Matrix* X);

/**
 * Dot product: sum(X .* Y)
 * Utilise cblas_ddot
 */
double matrix_dot(const Matrix* X, const Matrix* Y);

// ====================
// Opérations Element-wise (non-BLAS mais nécessaires)
// ====================

void matrix_add_scalar(Matrix* mat, double scalar);
void matrix_add_vector_broadcast(Matrix* mat, const Matrix* vec, int axis);
void matrix_relu(Matrix* mat);              // ReLU: max(0, x)
void matrix_relu_derivative(const Matrix* input, Matrix* grad);
void matrix_softmax(Matrix* mat);           // Softmax par ligne
void matrix_copy(const Matrix* src, Matrix* dst);
void matrix_hadamard(const Matrix* A, const Matrix* B, Matrix* C);  // element-wise multiply

// ====================
// Utilitaires
// ====================

void matrix_fill(Matrix* mat, double value);
void matrix_print(const Matrix* mat, const char* name);
double matrix_max(const Matrix* mat);
double matrix_min(const Matrix* mat);

#endif // MATRIX_H
