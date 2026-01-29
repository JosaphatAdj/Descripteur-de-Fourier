/**
 * fourier_naive.c - Implémentation NAÏVE des Descripteurs de Fourier
 * ===================================================================
 * 
 * Cette implémentation utilise des boucles C pures sans aucune 
 * optimisation BLAS pour servir de baseline dans les comparaisons.
 * 
 * ATTENTION: Code intentionnellement non optimisé!
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "include/fourier.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/**
 * Calcule les coefficients de Fourier de manière naïve.
 * Complexité: O(N * n_coefficients)
 */
int fourier_coefficients_naive(
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
    
    /* Boucle sur les indices de fréquence n */
    for (int n = -half_n; n <= half_n; n++) {
        double complex c_n = 0.0 + 0.0 * I;
        
        /* Boucle sur tous les points du contour */
        for (int k = 0; k < N; k++) {
            /* z_k = x_k + i * y_k */
            double complex z_k = contour->points[k].x + 
                                 contour->points[k].y * I;
            
            /* e^{-i * 2π * n * k / N} */
            double angle = -2.0 * M_PI * n * k / N;
            double complex exp_term = cos(angle) + sin(angle) * I;
            
            c_n += z_k * exp_term;
        }
        
        /* Normalisation par N */
        c_n /= N;
        
        /* Stocker dans le tableau (offset pour indices négatifs) */
        coefficients[n + half_n] = c_n;
    }
    
    return 0;
}


/**
 * Normalise les coefficients pour obtenir des descripteurs invariants.
 */
int normalize_descriptors_naive(
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
        /* Contour dégénéré, tous les descripteurs à 0 */
        for (int i = 0; i < n_coefficients - 1; i++) {
            descriptors[i] = 0.0;
        }
        return n_coefficients - 1;
    }
    
    /* Calcul des descripteurs normalisés */
    int desc_idx = 0;
    for (int i = 0; i < n_coefficients; i++) {
        if (i == center_idx) {
            /* Ignorer c_0 (translation) */
            continue;
        }
        descriptors[desc_idx] = cabs(coefficients[i]) / c1_magnitude;
        desc_idx++;
    }
    
    return desc_idx;
}


/**
 * Distance euclidienne naïve entre deux vecteurs.
 */
double distance_naive(const double* desc1, const double* desc2, int n) {
    if (!desc1 || !desc2 || n <= 0) {
        return -1.0;
    }
    
    double sum_squared = 0.0;
    
    for (int i = 0; i < n; i++) {
        double diff = desc1[i] - desc2[i];
        sum_squared += diff * diff;
    }
    
    return sqrt(sum_squared);
}


/**
 * Produit scalaire naïf (équivalent BLAS ddot).
 */
double dot_product_naive(const double* x, const double* y, int n) {
    if (!x || !y || n <= 0) {
        return 0.0;
    }
    
    double result = 0.0;
    
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    
    return result;
}


/**
 * Norme euclidienne naïve (équivalent BLAS dnrm2).
 */
double norm_naive(const double* x, int n) {
    if (!x || n <= 0) {
        return 0.0;
    }
    
    double sum_squared = 0.0;
    
    for (int i = 0; i < n; i++) {
        sum_squared += x[i] * x[i];
    }
    
    return sqrt(sum_squared);
}
