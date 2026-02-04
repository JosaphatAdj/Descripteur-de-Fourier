"""
Fallback: Import du wrapper Fourier

Si l'extension Cython n'est pas compilée, utilise le fallback Python pur.
"""
import numpy as np

class FourierWrapper:
    """Fallback Python pur si Cython non disponible."""
    
    def __init__(self, use_openblas=True):
        self.use_openblas = False  # Force Python fallback
        print("⚠ Wrapper Cython non disponible, utilisation du fallback Python")
    
    def compute_descriptors(self, contour, n_coefficients=32):
        """Calcul Python pur (FFT NumPy)."""
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
    
    def compute_descriptors_batch(self, contours_list, n_coefficients):
        """Batch fallback (boucle)."""
        return np.array([self.compute_descriptors(c, n_coefficients) for c in contours_list])
