/**
 * fourier.h - API des Descripteurs de Fourier avec OpenBLAS
 * ==========================================================
 * 
 * Ce header définit l'interface pour le calcul des descripteurs
 * de Fourier avec deux implémentations:
 *   - Naïve (boucles C pures) pour baseline
 *   - Optimisée (OpenBLAS) pour performance
 * 
 * Auteur: Projet OpenBLAS
 * Date: 2026
 */

#ifndef FOURIER_H
#define FOURIER_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * TYPES ET STRUCTURES
 * ============================================================ */

/**
 * Structure représentant un point 2D
 */
typedef struct {
    double x;
    double y;
} Point2D;

/**
 * Structure représentant un contour (liste de points)
 */
typedef struct {
    Point2D* points;    /* Tableau de points */
    size_t n_points;    /* Nombre de points */
} Contour;

/**
 * Structure pour les descripteurs de Fourier
 */
typedef struct {
    double* values;     /* Valeurs des descripteurs */
    size_t n_values;    /* Nombre de descripteurs */
} FourierDescriptors;

/**
 * Structure pour les résultats de benchmark
 */
typedef struct {
    double time_naive_ms;      /* Temps implémentation naïve (ms) */
    double time_openblas_ms;   /* Temps implémentation OpenBLAS (ms) */
    double speedup;            /* Ratio d'accélération */
    int n_points;              /* Nombre de points du contour */
    int n_coefficients;        /* Nombre de coefficients */
} BenchmarkResult;


/* ============================================================
 * IMPLÉMENTATION NAÏVE (sans OpenBLAS)
 * ============================================================ */

/**
 * Calcule les coefficients de Fourier de manière naïve.
 * 
 * Formule: c_n = (1/N) * Σ z_k * e^{-i*2π*n*k/N}
 * 
 * @param contour Contour d'entrée
 * @param coefficients Tableau pré-alloué pour les coefficients (complexes)
 * @param n_coefficients Nombre de coefficients à calculer
 * @return 0 si succès, -1 si erreur
 */
int fourier_coefficients_naive(
    const Contour* contour,
    double complex* coefficients,
    int n_coefficients
);

/**
 * Normalise les coefficients pour obtenir des descripteurs invariants.
 * Version naïve avec boucles.
 * 
 * @param coefficients Coefficients de Fourier
 * @param n_coefficients Nombre de coefficients
 * @param descriptors Tableau pré-alloué pour les descripteurs
 * @return Nombre de descripteurs calculés
 */
int normalize_descriptors_naive(
    const double complex* coefficients,
    int n_coefficients,
    double* descriptors
);

/**
 * Calcule la distance euclidienne entre deux descripteurs (naïf).
 * 
 * @param desc1 Premier ensemble de descripteurs
 * @param desc2 Deuxième ensemble de descripteurs
 * @param n Taille des descripteurs
 * @return Distance euclidienne
 */
double distance_naive(
    const double* desc1,
    const double* desc2,
    int n
);

/**
 * Produit scalaire naïf (équivalent à BLAS ddot).
 */
double dot_product_naive(const double* x, const double* y, int n);

/**
 * Norme euclidienne naïve (équivalent à BLAS dnrm2).
 */
double norm_naive(const double* x, int n);


/* ============================================================
 * IMPLÉMENTATION OPTIMISÉE (avec OpenBLAS)
 * ============================================================ */

/**
 * Calcule les coefficients de Fourier avec OpenBLAS.
 * Utilise les opérations vectorisées pour l'efficacité.
 * 
 * @param contour Contour d'entrée
 * @param coefficients Tableau pré-alloué pour les coefficients
 * @param n_coefficients Nombre de coefficients à calculer
 * @return 0 si succès, -1 si erreur
 */
int fourier_coefficients_openblas(
    const Contour* contour,
    double complex* coefficients,
    int n_coefficients
);

/**
 * Normalise les coefficients avec OpenBLAS.
 * Utilise cblas_dnrm2 pour les normes.
 * 
 * @param coefficients Coefficients de Fourier
 * @param n_coefficients Nombre de coefficients
 * @param descriptors Tableau pré-alloué pour les descripteurs
 * @return Nombre de descripteurs calculés
 */
int normalize_descriptors_openblas(
    const double complex* coefficients,
    int n_coefficients,
    double* descriptors
);

/**
 * Calcule la distance euclidienne avec OpenBLAS (dnrm2).
 * 
 * @param desc1 Premier ensemble de descripteurs
 * @param desc2 Deuxième ensemble de descripteurs
 * @param n Taille des descripteurs
 * @return Distance euclidienne
 */
double distance_openblas(
    const double* desc1,
    const double* desc2,
    int n
);

/**
 * Produit scalaire avec BLAS ddot.
 */
double dot_product_openblas(const double* x, const double* y, int n);

/**
 * Norme euclidienne avec BLAS dnrm2.
 */
double norm_openblas(const double* x, int n);


/* ============================================================
 * OPÉRATIONS BATCH (BLAS Level 3)
 * ============================================================ */

/**
 * Pré-calcule la matrice de Transformation de Fourier (W).
 * W est une matrice (n_coeffs x n_points) de complexes.
 * W[n, k] = (1/N) * exp(-i * 2*pi * n * k / N)
 * 
 * @param n_points Nombre de points par contour
 * @param n_coeffs Nombre de coefficients à conserver
 * @return Pointeur vers la matrice (allouée dynamiquement)
 */
double complex* precompute_dft_matrix(int n_points, int n_coeffs);

/**
 * Calcule les descripteurs pour un lot de contours en une seule opération (GEMM).
 * 
 * @param contours_batch Matrice de contours (batch_size x n_points). 
 *                       Stocké comme (batch_size * n_points) points contigus.
 * @param batch_size     Nombre de contours
 * @param n_points       Nombre de points par contour
 * @param dft_matrix     Matrice DFT pré-calculée (n_coeffs x n_points)
 * @param n_coeffs       Nombre de coefficients (lignes de dft_matrix)
 * @param output_coeffs  Buffer de sortie (batch_size x n_coeffs). 
 */
void fourier_batch_gemm(
    const double complex* contours_batch, 
    int batch_size, 
    int n_points,
    const double complex* dft_matrix, 
    int n_coeffs,
    double complex* output_coeffs
);

/**
 * Libère la matrice DFT.
 */
void free_dft_matrix(double complex* matrix);


/* ============================================================
 * UTILITAIRES
 * ============================================================ */

/**
 * Crée un contour à partir d'un tableau de coordonnées.
 * 
 * @param x Tableau des coordonnées X
 * @param y Tableau des coordonnées Y
 * @param n Nombre de points
 * @return Contour alloué (à libérer avec contour_free)
 */
Contour* contour_create(const double* x, const double* y, size_t n);

/**
 * Libère la mémoire d'un contour.
 */
void contour_free(Contour* contour);

/**
 * Génère un cercle de test.
 * 
 * @param n_points Nombre de points
 * @param radius Rayon du cercle
 * @return Contour du cercle
 */
Contour* contour_create_circle(int n_points, double radius);

/**
 * Génère un carré de test.
 */
Contour* contour_create_square(int n_points_per_side, double side_length);

/**
 * Alloue un tableau de descripteurs.
 */
FourierDescriptors* descriptors_create(size_t n);

/**
 * Libère un tableau de descripteurs.
 */
void descriptors_free(FourierDescriptors* desc);


/* ============================================================
 * BENCHMARKING
 * ============================================================ */

/**
 * Exécute un benchmark comparatif naïf vs OpenBLAS.
 * 
 * @param n_points Nombre de points du contour
 * @param n_coefficients Nombre de coefficients
 * @param n_iterations Nombre d'itérations pour moyenner
 * @return Résultats du benchmark
 */
BenchmarkResult benchmark_compare(
    int n_points,
    int n_coefficients,
    int n_iterations
);

/**
 * Affiche les informations de configuration OpenBLAS.
 */
void print_openblas_config(void);


#ifdef __cplusplus
}
#endif

#endif /* FOURIER_H */
