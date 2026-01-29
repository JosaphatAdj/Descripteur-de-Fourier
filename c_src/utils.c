/**
 * utils.c - Utilitaires et fonctions de benchmark
 * ================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <complex.h>
#include "include/fourier.h"


/**
 * Mesure le temps en millisecondes.
 */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}


/**
 * Exécute un benchmark comparatif.
 */
BenchmarkResult benchmark_compare(
    int n_points,
    int n_coefficients,
    int n_iterations
) {
    BenchmarkResult result = {0};
    result.n_points = n_points;
    result.n_coefficients = n_coefficients;
    
    /* Créer un contour de test (cercle) */
    Contour* contour = contour_create_circle(n_points, 1.0);
    if (!contour) {
        fprintf(stderr, "Erreur: impossible de créer le contour\n");
        return result;
    }
    
    /* Allouer les tableaux */
    double complex* coeffs = (double complex*)malloc(
        (n_coefficients + 1) * sizeof(double complex));
    double* descriptors = (double*)malloc(n_coefficients * sizeof(double));
    
    if (!coeffs || !descriptors) {
        fprintf(stderr, "Erreur: allocation mémoire\n");
        contour_free(contour);
        free(coeffs);
        free(descriptors);
        return result;
    }
    
    /* ========================
     * Benchmark NAÏF
     * ======================== */
    double start = get_time_ms();
    
    for (int iter = 0; iter < n_iterations; iter++) {
        fourier_coefficients_naive(contour, coeffs, n_coefficients);
        normalize_descriptors_naive(coeffs, n_coefficients + 1, descriptors);
    }
    
    double end = get_time_ms();
    result.time_naive_ms = (end - start) / n_iterations;
    
    /* ========================
     * Benchmark OPENBLAS
     * ======================== */
    start = get_time_ms();
    
    for (int iter = 0; iter < n_iterations; iter++) {
        fourier_coefficients_openblas(contour, coeffs, n_coefficients);
        normalize_descriptors_openblas(coeffs, n_coefficients + 1, descriptors);
    }
    
    end = get_time_ms();
    result.time_openblas_ms = (end - start) / n_iterations;
    
    /* Calculer le speedup */
    if (result.time_openblas_ms > 0) {
        result.speedup = result.time_naive_ms / result.time_openblas_ms;
    }
    
    /* Nettoyage */
    contour_free(contour);
    free(coeffs);
    free(descriptors);
    
    return result;
}


/**
 * Benchmark des opérations BLAS Level 1.
 */
void benchmark_blas_operations(int vector_size, int n_iterations) {
    printf("\n=== Benchmark BLAS Level 1 (taille=%d, iter=%d) ===\n",
           vector_size, n_iterations);
    
    /* Allouer les vecteurs */
    double* x = (double*)malloc(vector_size * sizeof(double));
    double* y = (double*)malloc(vector_size * sizeof(double));
    
    if (!x || !y) {
        fprintf(stderr, "Erreur allocation\n");
        free(x);
        free(y);
        return;
    }
    
    /* Initialiser avec des valeurs aléatoires */
    for (int i = 0; i < vector_size; i++) {
        x[i] = (double)rand() / RAND_MAX;
        y[i] = (double)rand() / RAND_MAX;
    }
    
    double start, end;
    volatile double result;  /* volatile pour éviter l'optimisation */
    
    /* --- ddot naïf --- */
    start = get_time_ms();
    for (int iter = 0; iter < n_iterations; iter++) {
        result = dot_product_naive(x, y, vector_size);
    }
    end = get_time_ms();
    double time_dot_naive = (end - start) / n_iterations;
    
    /* --- ddot OpenBLAS --- */
    start = get_time_ms();
    for (int iter = 0; iter < n_iterations; iter++) {
        result = dot_product_openblas(x, y, vector_size);
    }
    end = get_time_ms();
    double time_dot_openblas = (end - start) / n_iterations;
    
    /* --- dnrm2 naïf --- */
    start = get_time_ms();
    for (int iter = 0; iter < n_iterations; iter++) {
        result = norm_naive(x, vector_size);
    }
    end = get_time_ms();
    double time_nrm_naive = (end - start) / n_iterations;
    
    /* --- dnrm2 OpenBLAS --- */
    start = get_time_ms();
    for (int iter = 0; iter < n_iterations; iter++) {
        result = norm_openblas(x, vector_size);
    }
    end = get_time_ms();
    double time_nrm_openblas = (end - start) / n_iterations;
    
    /* Afficher les résultats */
    printf("\nddot (produit scalaire):\n");
    printf("  Naïf:     %.4f ms\n", time_dot_naive);
    printf("  OpenBLAS: %.4f ms\n", time_dot_openblas);
    printf("  Speedup:  %.2fx\n", time_dot_naive / time_dot_openblas);
    
    printf("\ndnrm2 (norme euclidienne):\n");
    printf("  Naïf:     %.4f ms\n", time_nrm_naive);
    printf("  OpenBLAS: %.4f ms\n", time_nrm_openblas);
    printf("  Speedup:  %.2fx\n", time_nrm_naive / time_nrm_openblas);
    
    (void)result;  /* Supprimer le warning unused */
    
    free(x);
    free(y);
}
