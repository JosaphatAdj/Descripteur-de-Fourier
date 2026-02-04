# cython: language_level=3
# distutils: language=c
# distutils: libraries=fourier openblas m
# distutils: library_dirs=python/lib
# distutils: include_dirs=c_src/include

"""
Wrapper Cython SIMPLIFIÉ pour libfourier
=========================================

Évite les types complexes C qui causent des problèmes avec Cython.
Utilise NumPy FFT vectorisé pour le batch processing.
"""

cimport numpy as np
import numpy as np


cdef class FourierWrapper:
    """Wrapper Cython simplifié."""
    
    cdef public bint use_openblas
    
    def __init__(self, use_openblas=True):
        self.use_openblas = use_openblas
        if use_openblas:
            print("✓ Wrapper Cython chargé (batch FFT vectorisé)")
    
    def compute_descriptors(self, np.ndarray[np.float64_t, ndim=2] contour, 
                           int n_coefficients=32):
        """
        Calcule descripteurs via NumPy FFT (optimisé).
        """
        cdef int N = contour.shape[0]
        cdef np.ndarray[np.complex128_t, ndim=1] z
        cdef np.ndarray[np.complex128_t, ndim=1] coefficients
        cdef np.ndarray[np.float64_t, ndim=1] magnitudes
        cdef np.ndarray[np.float64_t, ndim=1] descriptors
        cdef int center, half_n, center_idx
        cdef double total_magnitude
        
        # Représentation complexe
        z = contour[:, 0] + 1j * contour[:, 1]
        
        # FFT (NumPy utilise FFTW ou MKL si disponible)
        coefficients = np.fft.fft(z)
        coefficients = np.fft.fftshift(coefficients) / N
        
        # Extraire coefficients centraux
        center = N // 2
        half_n = n_coefficients // 2
        coefficients = coefficients[center - half_n : center + half_n + 1]
        
        # Magnitude
        magnitudes = np.abs(coefficients)
        
        # Exclure C0 (centre)
        center_idx = len(coefficients) // 2
        descriptors = np.concatenate([magnitudes[:center_idx], magnitudes[center_idx+1:]])
        
        # Normalisation L1
        total_magnitude = np.sum(descriptors)
        if total_magnitude > 1e-10:
            descriptors = descriptors / total_magnitude
        
        return descriptors
    
    def compute_descriptors_batch(self, list contours_list, int n_coefficients):
        """
        Batch processing VECTORISÉ avec NumPy FFT.
        
        Cette version utilise np.fft.fft sur axis=1 pour traiter
        tout le batch en une seule passe (beaucoup plus rapide qu'une boucle).
        """
        cdef int batch_size = len(contours_list)
        cdef int n_points
        cdef np.ndarray[np.complex128_t, ndim=2] batch_complex
        cdef np.ndarray[np.complex128_t, ndim=2] batch_fft
        cdef np.ndarray[np.float64_t, ndim=2] batch_mag
        cdef np.ndarray[np.float64_t, ndim=2] descriptors_matrix
        cdef int i, center, half_n, center_idx
        cdef np.ndarray[np.float64_t, ndim=2] temp_contour
        
        if batch_size == 0:
            return np.empty((0, n_coefficients))
        
        # Vérifier tailles
        n_points = contours_list[0].shape[0]
        
        # Stack en matrice (Batch x Points)
        batch_complex = np.zeros((batch_size, n_points), dtype=np.complex128)
        for i in range(batch_size):
            temp_contour = contours_list[i]
            if temp_contour.shape[0] != n_points:
                raise ValueError("Tous les contours doivent avoir la même taille")
            batch_complex[i, :] = temp_contour[:, 0] + 1j * temp_contour[:, 1]
        
        # FFT sur axis=1 (vectorisé - traite tout le batch d'un coup!)
        batch_fft = np.fft.fft(batch_complex, axis=1)
        batch_fft = np.fft.fftshift(batch_fft, axes=1) / n_points
        
        # Extraire coefficients centraux
        center = n_points // 2
        half_n = n_coefficients // 2
        batch_fft = batch_fft[:, center - half_n : center + half_n + 1]
        
        # Magnitude
        batch_mag = np.abs(batch_fft)
        
        # Exclure C0
        center_idx = batch_fft.shape[1] // 2
        descriptors_matrix = np.hstack([
            batch_mag[:, :center_idx],
            batch_mag[:, center_idx+1:]
        ])
        
        # Normalisation L1 par ligne
        sums = descriptors_matrix.sum(axis=1, keepdims=True)
        sums[sums < 1e-10] = 1.0
        
        return descriptors_matrix / sums
    
    def compute_distance(self, np.ndarray[np.float64_t, ndim=1] desc1,
                        np.ndarray[np.float64_t, ndim=1] desc2):
        """Distance euclidienne."""
        return np.linalg.norm(desc1 - desc2)
