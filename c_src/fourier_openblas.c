/**
 * fourier_openblas.c - Implémentation OPTIMISÉE avec OpenBLAS
 * ============================================================
 * 
 * Cette implémentation utilise OpenBLAS pour les opérations 
 * vectorielles (BLAS Level 1):
 *   - ddot: produit scalaire
 *   - dnrm2: norme euclidienne
 *   - daxpy: y = alpha*x + y
 * 
 * Avantages:
 *   - Instructions SIMD (SSE, AVX)
 *   - Optimisation cache
 *   - Multi-threading automatique
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <cblas.h>         /* OpenBLAS CBLAS interface */
#include "include/fourier.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/**
 * Affiche les informations de configuration OpenBLAS.
 */
void print_openblas_config(void) {
    printf("=== Configuration OpenBLAS ===\n");
    
    /* Nombre de threads */
    int num_threads = openblas_get_num_threads();
    printf("Nombre de threads: %d\n", num_threads);
    
    /* Type de CPU */
    char* corename = openblas_get_corename();
    printf("Type de CPU: %s\n", corename);
    
    /* Configuration */
    int parallel = openblas_get_parallel();
    printf("Mode parallèle: %d (0=seq, 1=pthreads, 2=openmp)\n", parallel);
    
    printf("==============================\n");
}


/**
 * Calcule les coefficients de Fourier avec OpenBLAS.
 * 
 * Optimisations:
 *   - Pré-calcul des exponentielles
 *   - Utilisation de cblas_zdotu pour les produits complexes
 */
int fourier_coefficients_openblas(
    const Contour* contour,
    double complex* coefficients,
    int n_coefficients
) {
    if (!contour || !coefficients || n_coefficients <= 0) {
        return -1;
    }
    
    int N = (int)contour->n_points;
    if (N == 0) {
        return -1;
    }
    
    int half_n = n_coefficients / 2;
    
    /* Convertir le contour en tableau de complexes */
    double complex* z = (double complex*)malloc(N * sizeof(double complex));
    if (!z) {
        return -1;
    }
    
    for (int k = 0; k < N; k++) {
        z[k] = contour->points[k].x + contour->points[k].y * I;
    }
    
    /* Pré-allouer le tableau des exponentielles */
    double complex* exp_table = (double complex*)malloc(N * sizeof(double complex));
    if (!exp_table) {
        free(z);
        return -1;
    }
    
    /* Calculer les coefficients */
    for (int n = -half_n; n <= half_n; n++) {
        /* Construire le tableau d'exponentielles pour cette fréquence */
        for (int k = 0; k < N; k++) {
            double angle = -2.0 * M_PI * n * k / N;
            exp_table[k] = cos(angle) + sin(angle) * I;
        }
        
        /* 
         * Produit scalaire complexe: c_n = (1/N) * Σ z_k * exp_k
         * Utilise cblas_zdotu (BLAS Level 1)
         */
        double complex c_n = 0.0;
        
        /* OpenBLAS zdotu: produit scalaire de vecteurs complexes */
        cblas_zdotu_sub(N, z, 1, exp_table, 1, &c_n);
        
        c_n /= N;
        coefficients[n + half_n] = c_n;
    }
    
    free(z);
    free(exp_table);
    
    return 0;
}


/**
 * Normalise les coefficients avec OpenBLAS.
 * Utilise cblas_dnrm2 pour les normes.
 */
int normalize_descriptors_openblas(
    const double complex* coefficients,
    int n_coefficients,
    double* descriptors
) {
    if (!coefficients || !descriptors || n_coefficients < 2) {
        return -1;
    }
    
    int center_idx = n_coefficients / 2;
    int c1_idx = center_idx + 1;
    
    if (c1_idx >= n_coefficients) {
        return -1;
    }
    
    /* |c_1| pour la normalisation */
    double c1_magnitude = cabs(coefficients[c1_idx]);
    
    if (c1_magnitude < 1e-10) {
        /* Utilise cblas_dscal pour mettre à zéro rapidement */
        memset(descriptors, 0, (n_coefficients - 1) * sizeof(double));
        return n_coefficients - 1;
    }
    
    /* Extraire les magnitudes (vectorisable) */
    double* magnitudes = (double*)malloc(n_coefficients * sizeof(double));
    if (!magnitudes) {
        return -1;
    }
    
    for (int i = 0; i < n_coefficients; i++) {
        magnitudes[i] = cabs(coefficients[i]);
    }
    
    /* Normaliser et copier (excluant c_0) */
    double inv_c1 = 1.0 / c1_magnitude;
    int desc_idx = 0;
    
    for (int i = 0; i < n_coefficients; i++) {
        if (i == center_idx) continue;
        descriptors[desc_idx] = magnitudes[i] * inv_c1;
        desc_idx++;
    }
    
    free(magnitudes);
    return desc_idx;
}


/**
 * Distance euclidienne avec OpenBLAS.
 * Utilise cblas_daxpy et cblas_dnrm2.
 */
double distance_openblas(const double* desc1, const double* desc2, int n) {
    if (!desc1 || !desc2 || n <= 0) {
        return -1.0;
    }
    
    /* Allouer un vecteur temporaire pour la différence */
    double* diff = (double*)malloc(n * sizeof(double));
    if (!diff) {
        return -1.0;
    }
    
    /* Copier desc1 dans diff */
    memcpy(diff, desc1, n * sizeof(double));
    
    /* diff = desc1 - desc2 = diff + (-1.0) * desc2 */
    cblas_daxpy(n, -1.0, desc2, 1, diff, 1);
    
    /* Norme de la différence avec BLAS dnrm2 */
    double result = cblas_dnrm2(n, diff, 1);
    
    free(diff);
    return result;
}


/**
 * Produit scalaire avec BLAS ddot.
 */
double dot_product_openblas(const double* x, const double* y, int n) {
    if (!x || !y || n <= 0) {
        return 0.0;
    }
    
    /* BLAS Level 1: ddot */
    return cblas_ddot(n, x, 1, y, 1);
}


/**
 * Norme euclidienne avec BLAS dnrm2.
 */
double norm_openblas(const double* x, int n) {
    if (!x || n <= 0) {
        return 0.0;
    }
    
    /* BLAS Level 1: dnrm2 */
    return cblas_dnrm2(n, x, 1);
}
