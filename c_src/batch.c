#include "fourier.h"
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Pré-calcule la matrice de Transformation de Fourier (W).
 * W[n, k] = (1/N) * exp(-i * 2*pi * n * k / N)
 */
double complex* precompute_dft_matrix(int n_points, int n_coeffs) {
    double complex* W = (double complex*)malloc(n_coeffs * n_points * sizeof(double complex));
    if (!W) return NULL;

    for (int n = 0; n < n_coeffs; n++) {
        // Pour centrer les coefficients (n allant de -n_coeffs/2 à +n_coeffs/2 approx)
        // Mais ici on simplifie en stockant les "n" indices correspondant à la logique de numpy.fft/shift
        // Ou plus simplement: on suit la formule standard pour les n_coeffs premiers coefficients.
        // ATTENTION: Pour comparer avec l'implémentation naïve/numpy qui fait un fftshift,
        // il faut décider quels indices "n" on calcule.
        // Option simple: Calculer 0 à n_coeffs-1. (Basse fréquence = début).
        // Mais Fourier descriptors utilise souvent les basses fréquences centrées.
        // Pour matcher `analyze_casting.py` (qui prend indices centrés après fftshift),
        // on va générer les indices: -(M/2) ... 0 ... +(M/2).
        
        int freq_idx = n - (n_coeffs / 2); // De -16 à +16 par exemple
        
        for (int k = 0; k < n_points; k++) {
            double angle = -2.0 * M_PI * freq_idx * k / (double)n_points;
            W[n * n_points + k] = (cos(angle) + I * sin(angle)) / (double)n_points;
        }
    }
    return W;
}

void free_dft_matrix(double complex* matrix) {
    if (matrix) free(matrix);
}

/*
 * Calcule C = X * W^T
 * X: (batch_size, n_points)
 * W: (n_coeffs, n_points)
 * C: (batch_size, n_coeffs)
 */
void fourier_batch_gemm(
    const double complex* contours_batch, 
    int batch_size, 
    int n_points,
    const double complex* dft_matrix, 
    int n_coeffs,
    double complex* output_coeffs
) {
    double complex alpha = 1.0 + 0.0 * I;
    double complex beta = 0.0 + 0.0 * I;

    // cblas_zgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    // C = alpha * A * B + beta * C
    // Nous voulons: Output = Contours * Transpose(DFT)
    // M = batch_size
    // N = n_coeffs
    // K = n_points
    // A = contours_batch (M x K)
    // B = dft_matrix (N x K) -> On veut B^T (K x N), donc on utilise TransB
    
    cblas_zgemm(
        CblasRowMajor, 
        CblasNoTrans, // A n'est pas transposé
        CblasTrans,   // B est transposé (on lit W comme W^T)
        batch_size,   // M
        n_coeffs,     // N
        n_points,     // K
        &alpha,       
        contours_batch, 
        n_points,     // lda (stride de A)
        dft_matrix, 
        n_points,     // ldb (stride de B, car B est stocké (coeffs x points))
        &beta, 
        output_coeffs, 
        n_coeffs      // ldc (stride de C)
    );
}
