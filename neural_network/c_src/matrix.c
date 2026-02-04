/**
 * Implémentation des opérations matricielles avec OpenBLAS
 * =========================================================
 * 
 * Wrapper pour maximiser l'utilisation des routines BLAS optimisées.
 */

#include "matrix.h"
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================
// Allocation/Libération
// ============================================

Matrix* matrix_create(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)malloc(rows * cols * sizeof(double));
    
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    
    return mat;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        if (mat->data) free(mat->data);
        free(mat);
    }
}

Matrix* matrix_zeros(int rows, int cols) {
    Matrix* mat = matrix_create(rows, cols);
    if (mat) memset(mat->data, 0, rows * cols * sizeof(double));
    return mat;
}

Matrix* matrix_ones(int rows, int cols) {
    Matrix* mat = matrix_create(rows, cols);
    if (mat) {
        matrix_fill(mat, 1.0);
    }
    return mat;
}

Matrix* matrix_random_normal(int rows, int cols, double mean, double std) {
    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;
    
    // Box-Muller transform pour distribution normale
    for (int i = 0; i < rows * cols; i += 2) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        mat->data[i] = mean + std * z0;
        
        if (i + 1 < rows * cols) {
            double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
            mat->data[i + 1] = mean + std * z1;
        }
    }
    
    return mat;
}

// ============================================
// BLAS Level 3: GEMM (LE PLUS IMPORTANT!)
// ============================================

void matrix_multiply(const Matrix* A, const Matrix* B, Matrix* C,
                    double alpha, double beta,
                    int transpose_A, int transpose_B) {
    /**
     * C = alpha * op(A) * op(B) + beta * C
     * 
     * Utilise cblas_dgemm - routine la plus optimisée d'OpenBLAS
     * C'est ici que 90%+ du temps CPU sera passé !
     */
    
    int M = transpose_A ? A->cols : A->rows;
    int N = transpose_B ? B->rows : B->cols;
    int K = transpose_A ? A->rows : A->cols;
    
    // Vérification dimensions
    int K_check = transpose_B ? B->cols : B->rows;
    if (K != K_check || C->rows != M || C->cols != N) {
        fprintf(stderr, "Erreur dimensions GEMM: (%d×%d) × (%d×%d) → (%d×%d)\n",
                M, K, K, N, C->rows, C->cols);
        return;
    }
    
    CBLAS_TRANSPOSE trans_A = transpose_A ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = transpose_B ? CblasTrans : CblasNoTrans;
    
    // GEMM OpenBLAS - c'est ici que la magie opère!
    cblas_dgemm(
        CblasRowMajor,
        trans_A, trans_B,
        M, N, K,
        alpha,
        A->data, A->cols,
        B->data, B->cols,
        beta,
        C->data, C->cols
    );
}

// ============================================
// BLAS Level 1
// ============================================

void matrix_axpy(double alpha, const Matrix* X, Matrix* Y) {
    /**
     * Y = Y + alpha*X
     * Utilise cblas_daxpy
     */
    int n = X->rows * X->cols;
    cblas_daxpy(n, alpha, X->data, 1, Y->data, 1);
}

void matrix_scale(double alpha, Matrix* X) {
    /**
     * X = alpha*X
     * Utilise cblas_dscal
     */
    int n = X->rows * X->cols;
    cblas_dscal(n, alpha, X->data, 1);
}

double matrix_dot(const Matrix* X, const Matrix* Y) {
    /**
     * sum(X .* Y)
     * Utilise cblas_ddot
     */
    int n = X->rows * X->cols;
    return cblas_ddot(n, X->data, 1, Y->data, 1);
}

// ============================================
// Opérations Element-wise
// ============================================

void matrix_add_scalar(Matrix* mat, double scalar) {
    int n = mat->rows * mat->cols;
    for (int i = 0; i < n; i++) {
        mat->data[i] += scalar;
    }
}

void matrix_add_vector_broadcast(Matrix* mat, const Matrix* vec, int axis) {
    /**
     * Broadcasting addition (comme NumPy)
     * axis=0: ajoute vec à chaque ligne
     * axis=1: ajoute vec à chaque colonne
     */
    if (axis == 1) {
        // Ajouter vec (1 × cols) à chaque ligne
        for (int i = 0; i < mat->rows; i++) {
            cblas_daxpy(mat->cols, 1.0, vec->data, 1, 
                       mat->data + i * mat->cols, 1);
        }
    } else {
        // Ajouter vec (rows × 1) à chaque colonne
        for (int i = 0; i < mat->rows; i++) {
            double val = vec->data[i];
            for (int j = 0; j < mat->cols; j++) {
                mat->data[i * mat->cols + j] += val;
            }
        }
    }
}

void matrix_relu(Matrix* mat) {
    /**
     * ReLU activation: max(0, x)
     */
    int n = mat->rows * mat->cols;
    for (int i = 0; i < n; i++) {
        if (mat->data[i] < 0.0) {
            mat->data[i] = 0.0;
        }
    }
}

void matrix_relu_derivative(const Matrix* input, Matrix* grad) {
    /**
     * ReLU gradient: grad *= (input > 0)
     */
    int n = input->rows * input->cols;
    for (int i = 0; i < n; i++) {
        if (input->data[i] <= 0.0) {
            grad->data[i] = 0.0;
        }
    }
}

void matrix_softmax(Matrix* mat) {
    /**
     * Softmax par ligne (stable numériquement)
     */
    for (int i = 0; i < mat->rows; i++) {
        double* row = mat->data + i * mat->cols;
        
        // Max pour stabilité
        double max_val = row[0];
        for (int j = 1; j < mat->cols; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        
        // Exp et somme
        double sum = 0.0;
        for (int j = 0; j < mat->cols; j++) {
            row[j] = exp(row[j] - max_val);
            sum += row[j];
        }
        
        // Normalisation
        for (int j = 0; j < mat->cols; j++) {
            row[j] /= sum;
        }
    }
}

void matrix_copy(const Matrix* src, Matrix* dst) {
    if (src->rows != dst->rows || src->cols != dst->cols) {
        fprintf(stderr, "Erreur copy: dimensions incompatibles\n");
        return;
    }
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(double));
}

void matrix_hadamard(const Matrix* A, const Matrix* B, Matrix* C) {
    /**
     * Element-wise multiply: C = A .* B
     */
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
}

// ============================================
// Utilitaires
// ============================================

void matrix_fill(Matrix* mat, double value) {
    int n = mat->rows * mat->cols;
    for (int i = 0; i < n; i++) {
        mat->data[i] = value;
    }
}

void matrix_print(const Matrix* mat, const char* name) {
    printf("%s (%d × %d):\n", name, mat->rows, mat->cols);
    
    int max_rows = (mat->rows > 5) ? 5 : mat->rows;
    int max_cols = (mat->cols > 5) ? 5 : mat->cols;
    
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            printf("%8.4f ", mat->data[i * mat->cols + j]);
        }
        if (mat->cols > 5) printf("...");
        printf("\n");
    }
    if (mat->rows > 5) printf("...\n");
    printf("\n");
}

double matrix_max(const Matrix* mat) {
    int n = mat->rows * mat->cols;
    double max_val = mat->data[0];
    for (int i = 1; i < n; i++) {
        if (mat->data[i] > max_val) max_val = mat->data[i];
    }
    return max_val;
}

double matrix_min(const Matrix* mat) {
    int n = mat->rows * mat->cols;
    double min_val = mat->data[0];
    for (int i = 1; i < n; i++) {
        if (mat->data[i] < min_val) min_val = mat->data[i];
    }
    return min_val;
}
