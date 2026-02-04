# cython: language_level=3
# distutils: language=c
# distutils: libraries=fourier openblas m
# distutils: library_dirs=python/lib
# distutils: include_dirs=c_src/include

"""
Wrapper Cython pour la bibliothèque C libfourier
================================================

Utilise Cython pour un accès direct aux types complexes C.
"""

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t

# Déclarations externes C
cdef extern from "complex.h":
    ctypedef double complex double_complex

cdef extern from "fourier.h":
    # Structures
    ctypedef struct Point2D:
        double x
        double y
    
    ctypedef struct Contour:
        Point2D* points
        size_t n_points
    
    # Gestion des contours
    Contour* contour_create(const double* x, const double* y, size_t n)
    void contour_free(Contour* contour)
    
    # Fonctions Fourier (OpenBLAS)
    int fourier_coefficients_openblas(const Contour* contour, 
                                      double_complex* coefficients,
                                      int n_coefficients)
    int normalize_descriptors_openblas(const double_complex* coefficients,
                                       int n_coefficients,
                                       double* descriptors)
    
    # Fonctions batch
    double_complex* precompute_dft_matrix(int n_points, int n_coeffs)
    void fourier_batch_gemm(const double_complex* contours_batch,
                           int batch_size,
                           int n_points,
                           const double_complex* dft_matrix,
                           int n_coeffs,
                           double_complex* output_coeffs)
    void free_dft_matrix(double_complex* matrix)
    
    # Distance/norms
    double distance_openblas(const double* desc1, const double* desc2, int n)
    double norm_openblas(const double* x, int n)


cdef class FourierWrapper:
    """Wrapper Cython pour les descripteurs de Fourier."""
    
    cdef public bint use_openblas
    
    def __init__(self, use_openblas=True):
        self.use_openblas = use_openblas
    
    def compute_descriptors(self, np.ndarray[np.float64_t, ndim=2] contour, 
                           int n_coefficients=32):
        """
        Calcule les descripteurs de Fourier d'un contour.
        
        Args:
            contour: Array numpy (N, 2) avec points (x, y)
            n_coefficients: Nombre de coefficients
            
        Returns:
            Array numpy des descripteurs normalisés
        """
        cdef int n_points = contour.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] x = np.ascontiguousarray(contour[:, 0])
        cdef np.ndarray[np.float64_t, ndim=1] y = np.ascontiguousarray(contour[:, 1])
        
        # Créer le contour C
        cdef Contour* c_contour = contour_create(&x[0], &y[0], n_points)
        
        # Allouer coefficients et descripteurs
        cdef np.ndarray[np.complex128_t, ndim=1] coefficients = np.zeros(n_coefficients + 1, dtype=np.complex128)
        cdef np.ndarray[np.float64_t, ndim=1] descriptors = np.zeros(n_coefficients, dtype=np.float64)
        
        cdef int n_desc
        
        try:
            if self.use_openblas:
                fourier_coefficients_openblas(c_contour, 
                                              <double_complex*>&coefficients[0],
                                              n_coefficients)
                n_desc = normalize_descriptors_openblas(<double_complex*>&coefficients[0],
                                                        n_coefficients + 1,
                                                        &descriptors[0])
            else:
                # Fallback Python
                return self._compute_python(contour, n_coefficients)
            
            return descriptors[:n_desc]
        finally:
            contour_free(c_contour)
    
    def compute_descriptors_batch(self, list contours_list, int n_coefficients):
        """
        Calcule les descripteurs pour un batch de contours (GEMM).
        
        Args:
            contours_list: Liste de contours (N, 2)
            n_coefficients: Nombre de coefficients
            
        Returns:
            Matrice (BatchSize, n_coefficients)
        """
        if not self.use_openblas:
            # Fallback
            return np.array([self.compute_descriptors(c, n_coefficients) for c in contours_list])
        
        cdef int batch_size = len(contours_list)
        if batch_size == 0:
            return np.empty((0, n_coefficients))
        
        cdef int n_points = contours_list[0].shape[0]
        
        # Préparer input batch
        cdef np.ndarray[np.complex128_t, ndim=1] input_data = np.zeros(batch_size * n_points, dtype=np.complex128)
        cdef np.ndarray[np.complex128_t, ndim=1] output_coeffs = np.zeros(batch_size * n_coefficients, dtype=np.complex128)
        
        cdef int i
        cdef np.ndarray[np.float64_t, ndim=2] c
        
        for i in range(batch_size):
            c = contours_list[i]
            if c.shape[0] != n_points:
                raise ValueError("Tous les contours doivent avoir la même taille")
            input_data[i*n_points:(i+1)*n_points] = c[:, 0] + 1j * c[:, 1]
        
        # Appeler C (GEMM)
        cdef double_complex* W_ptr = precompute_dft_matrix(n_points, n_coefficients)
        
        try:
            fourier_batch_gemm(<double_complex*>&input_data[0],
                              batch_size,
                              n_points,
                              W_ptr,
                              n_coefficients,
                              <double_complex*>&output_coeffs[0])
        finally:
            free_dft_matrix(W_ptr)
        
        # Post-traitement (normalisation L1)
        cdef np.ndarray[np.complex128_t, ndim=2] coeffs_matrix = output_coeffs.reshape(batch_size, n_coefficients)
        cdef np.ndarray[np.float64_t, ndim=2] magnitudes = np.abs(coeffs_matrix)
        
        # Normalisation
        cdef np.ndarray[np.float64_t, ndim=2] sums = magnitudes.sum(axis=1, keepdims=True)
        sums[sums < 1e-10] = 1.0
        
        return magnitudes / sums
    
    def _compute_python(self, contour, n_coefficients):
        """Fallback Python."""
        N = len(contour)
        z = contour[:, 0] + 1j * contour[:, 1]
        coefficients = np.fft.fft(z)
        coefficients = np.fft.fftshift(coefficients) / N
        
        center = N // 2
        half_n = n_coefficients // 2
        coeffs = coefficients[center - half_n : center + half_n + 1]
        
        magnitudes = np.abs(coeffs)
        center_idx = len(coeffs) // 2
        mask = np.ones(len(coeffs), dtype=bool)
        mask[center_idx] = False
        
        descriptors = magnitudes[mask]
        total_magnitude = np.sum(descriptors)
        
        if total_magnitude > 1e-10:
            descriptors = descriptors / total_magnitude
        
        return descriptors
    
    def compute_distance(self, np.ndarray[np.float64_t, ndim=1] desc1,
                        np.ndarray[np.float64_t, ndim=1] desc2):
        """Calcule la distance euclidienne."""
        if not self.use_openblas:
            return np.linalg.norm(desc1 - desc2)
        
        return distance_openblas(&desc1[0], &desc2[0], len(desc1))
