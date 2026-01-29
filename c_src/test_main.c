/**
 * test_main.c - Programme principal de test
 * ==========================================
 * 
 * Ce programme teste les implémentations et compare
 * les performances naïf vs OpenBLAS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "include/fourier.h"

/* Déclaration externe du benchmark BLAS */
extern void benchmark_blas_operations(int vector_size, int n_iterations);


void test_basic_functionality(void) {
    printf("\n========================================\n");
    printf("TEST 1: Fonctionnalité de base\n");
    printf("========================================\n");
    
    /* Créer un cercle */
    int n_points = 100;
    Contour* circle = contour_create_circle(n_points, 1.0);
    
    if (!circle) {
        printf("ÉCHEC: Impossible de créer le cercle\n");
        return;
    }
    printf("✓ Cercle créé avec %zu points\n", circle->n_points);
    
    /* Calculer les coefficients (naïf) */
    int n_coefficients = 32;
    double complex* coeffs = (double complex*)malloc(
        (n_coefficients + 1) * sizeof(double complex));
    
    int ret = fourier_coefficients_naive(circle, coeffs, n_coefficients);
    if (ret == 0) {
        printf("✓ Coefficients naïfs calculés\n");
    } else {
        printf("✗ Erreur calcul naïf\n");
    }
    
    /* Calculer les coefficients (OpenBLAS) */
    ret = fourier_coefficients_openblas(circle, coeffs, n_coefficients);
    if (ret == 0) {
        printf("✓ Coefficients OpenBLAS calculés\n");
    } else {
        printf("✗ Erreur calcul OpenBLAS\n");
    }
    
    /* Normaliser */
    double* descriptors = (double*)malloc(n_coefficients * sizeof(double));
    int n_desc = normalize_descriptors_naive(coeffs, n_coefficients + 1, descriptors);
    printf("✓ %d descripteurs normalisés\n", n_desc);
    
    /* Afficher quelques descripteurs */
    printf("\nPremiers descripteurs (cercle):\n");
    for (int i = 0; i < 5 && i < n_desc; i++) {
        printf("  FD[%d] = %.6f\n", i, descriptors[i]);
    }
    
    /* Nettoyage */
    contour_free(circle);
    free(coeffs);
    free(descriptors);
}


void test_blas_operations(void) {
    printf("\n========================================\n");
    printf("TEST 2: Opérations BLAS\n");
    printf("========================================\n");
    
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    int n = 5;
    
    /* Produit scalaire */
    double dot_naive = dot_product_naive(x, y, n);
    double dot_blas = dot_product_openblas(x, y, n);
    
    printf("Produit scalaire [1,2,3,4,5] · [5,4,3,2,1]:\n");
    printf("  Naïf:     %.2f\n", dot_naive);
    printf("  OpenBLAS: %.2f\n", dot_blas);
    printf("  Attendu:  35.00\n");
    
    if (dot_naive == 35.0 && dot_blas == 35.0) {
        printf("  ✓ OK\n");
    } else {
        printf("  ✗ ERREUR\n");
    }
    
    /* Norme */
    double norm_naive_val = norm_naive(x, n);
    double norm_blas_val = norm_openblas(x, n);
    double expected_norm = 7.416198;  /* sqrt(1+4+9+16+25) */
    
    printf("\nNorme de [1,2,3,4,5]:\n");
    printf("  Naïf:     %.6f\n", norm_naive_val);
    printf("  OpenBLAS: %.6f\n", norm_blas_val);
    printf("  Attendu:  %.6f\n", expected_norm);
    
    if (norm_naive_val > 7.41 && norm_naive_val < 7.42) {
        printf("  ✓ OK\n");
    } else {
        printf("  ✗ ERREUR\n");
    }
}


void test_benchmark(void) {
    printf("\n========================================\n");
    printf("TEST 3: Benchmark Naïf vs OpenBLAS\n");
    printf("========================================\n");
    
    /* Afficher la config OpenBLAS */
    print_openblas_config();
    
    /* Tests avec différentes tailles */
    int sizes[] = {100, 500, 1000, 5000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("\n%-10s %-12s %-12s %-10s\n", 
           "Points", "Naïf (ms)", "OpenBLAS (ms)", "Speedup");
    printf("--------------------------------------------\n");
    
    for (int i = 0; i < n_sizes; i++) {
        BenchmarkResult result = benchmark_compare(sizes[i], 32, 100);
        
        printf("%-10d %-12.4f %-12.4f %-10.2fx\n",
               result.n_points,
               result.time_naive_ms,
               result.time_openblas_ms,
               result.speedup);
    }
    
    /* Benchmark des opérations BLAS */
    benchmark_blas_operations(10000, 10000);
}


int main(void) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  Descripteurs de Fourier avec OpenBLAS   ║\n");
    printf("║           Tests et Benchmarks            ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    
    test_basic_functionality();
    test_blas_operations();
    test_benchmark();
    
    printf("\n========================================\n");
    printf("TESTS TERMINÉS\n");
    printf("========================================\n");
    
    return 0;
}
