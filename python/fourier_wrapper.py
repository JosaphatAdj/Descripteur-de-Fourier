"""
Wrapper Python pour la bibliothèque C libfourier
================================================

Ce module utilise CFFI pour appeler les fonctions C
optimisées avec OpenBLAS.
"""

import os
import numpy as np
from cffi import FFI
from pathlib import Path

# Chemin vers la bibliothèque partagée
LIB_DIR = Path(__file__).parent / "lib"
LIB_PATH = LIB_DIR / "libfourier.so"

# Définition de l'interface C
ffi = FFI()

ffi.cdef("""
    /* Types */
    typedef struct {
        double x;
        double y;
    } Point2D;
    
    typedef struct {
        Point2D* points;
        size_t n_points;
    } Contour;
    
    typedef struct {
        double* values;
        size_t n_values;
    } FourierDescriptors;
    
    typedef struct {
        double time_naive_ms;
        double time_openblas_ms;
        double speedup;
        int n_points;
        int n_coefficients;
    } BenchmarkResult;
    
    /* Fonctions de gestion des contours */
    Contour* contour_create(const double* x, const double* y, size_t n);
    void contour_free(Contour* contour);
    Contour* contour_create_circle(int n_points, double radius);
    Contour* contour_create_square(int n_points_per_side, double side_length);
    
    /* Descripteurs */
    FourierDescriptors* descriptors_create(size_t n);
    void descriptors_free(FourierDescriptors* desc);
    
    /* Fonctions naïves */
    int fourier_coefficients_naive(
        const Contour* contour,
        double complex* coefficients,
        int n_coefficients
    );
    int normalize_descriptors_naive(
        const double complex* coefficients,
        int n_coefficients,
        double* descriptors
    );
    double distance_naive(const double* desc1, const double* desc2, int n);
    double dot_product_naive(const double* x, const double* y, int n);
    double norm_naive(const double* x, int n);
    
    /* Fonctions OpenBLAS */
    int fourier_coefficients_openblas(
        const Contour* contour,
        double complex* coefficients,
        int n_coefficients
    );
    int normalize_descriptors_openblas(
        const double complex* coefficients,
        int n_coefficients,
        double* descriptors
    );
    double distance_openblas(const double* desc1, const double* desc2, int n);
    double dot_product_openblas(const double* x, const double* y, int n);
    double norm_openblas(const double* x, int n);
    
    /* Benchmark */
    BenchmarkResult benchmark_compare(int n_points, int n_coefficients, int n_iterations);
    void print_openblas_config(void);
""")

# Charger la bibliothèque
_lib = None

def _load_library():
    """Charge la bibliothèque partagée."""
    global _lib
    if _lib is None:
        if not LIB_PATH.exists():
            raise FileNotFoundError(
                f"Bibliothèque non trouvée: {LIB_PATH}\n"
                "Exécutez 'make docker-build' pour compiler le code C."
            )
        _lib = ffi.dlopen(str(LIB_PATH))
    return _lib

def get_lib():
    """Retourne la bibliothèque chargée."""
    return _load_library()


class FourierWrapper:
    """
    Classe wrapper pour les fonctions de descripteurs de Fourier.
    """
    
    def __init__(self, use_openblas=True):
        """
        Args:
            use_openblas: Si True, utilise l'implémentation OpenBLAS.
                         Si False, utilise l'implémentation naïve.
        """
        self.lib = get_lib()
        self.use_openblas = use_openblas
    
    def compute_descriptors(self, contour: np.ndarray, n_coefficients: int = 32) -> np.ndarray:
        """
        Calcule les descripteurs de Fourier d'un contour.
        
        Args:
            contour: Array numpy de shape (N, 2) avec les points (x, y)
            n_coefficients: Nombre de coefficients
            
        Returns:
            Array numpy des descripteurs normalisés
        """
        n_points = len(contour)
        
        # Préparer les données
        x = np.ascontiguousarray(contour[:, 0], dtype=np.float64)
        y = np.ascontiguousarray(contour[:, 1], dtype=np.float64)
        
        x_ptr = ffi.cast("double*", x.ctypes.data)
        y_ptr = ffi.cast("double*", y.ctypes.data)
        
        # Créer le contour C
        c_contour = self.lib.contour_create(x_ptr, y_ptr, n_points)
        
        try:
            # Allouer les coefficients
            coefficients = np.zeros(n_coefficients + 1, dtype=np.complex128)
            coeffs_ptr = ffi.cast("double complex*", coefficients.ctypes.data)
            
            # Allouer les descripteurs
            descriptors = np.zeros(n_coefficients, dtype=np.float64)
            desc_ptr = ffi.cast("double*", descriptors.ctypes.data)
            
            # Calculer
            if self.use_openblas:
                self.lib.fourier_coefficients_openblas(c_contour, coeffs_ptr, n_coefficients)
                n_desc = self.lib.normalize_descriptors_openblas(
                    coeffs_ptr, n_coefficients + 1, desc_ptr)
            else:
                self.lib.fourier_coefficients_naive(c_contour, coeffs_ptr, n_coefficients)
                n_desc = self.lib.normalize_descriptors_naive(
                    coeffs_ptr, n_coefficients + 1, desc_ptr)
            
            return descriptors[:n_desc]
            
        finally:
            self.lib.contour_free(c_contour)
    
    def compute_distance(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Calcule la distance entre deux descripteurs."""
        n = len(desc1)
        assert len(desc2) == n, "Les descripteurs doivent avoir la même taille"
        
        d1 = np.ascontiguousarray(desc1, dtype=np.float64)
        d2 = np.ascontiguousarray(desc2, dtype=np.float64)
        
        d1_ptr = ffi.cast("double*", d1.ctypes.data)
        d2_ptr = ffi.cast("double*", d2.ctypes.data)
        
        if self.use_openblas:
            return self.lib.distance_openblas(d1_ptr, d2_ptr, n)
        else:
            return self.lib.distance_naive(d1_ptr, d2_ptr, n)
    
    def dot_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """Produit scalaire."""
        n = len(x)
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        x_ptr = ffi.cast("double*", x.ctypes.data)
        y_ptr = ffi.cast("double*", y.ctypes.data)
        
        if self.use_openblas:
            return self.lib.dot_product_openblas(x_ptr, y_ptr, n)
        else:
            return self.lib.dot_product_naive(x_ptr, y_ptr, n)
    
    def norm(self, x: np.ndarray) -> float:
        """Norme euclidienne."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        x_ptr = ffi.cast("double*", x.ctypes.data)
        
        if self.use_openblas:
            return self.lib.norm_openblas(x_ptr, len(x))
        else:
            return self.lib.norm_naive(x_ptr, len(x))


def run_benchmark(n_points: int = 1000, n_coefficients: int = 32, 
                  n_iterations: int = 100) -> dict:
    """
    Exécute un benchmark comparatif naïf vs OpenBLAS.
    
    Returns:
        Dictionnaire avec les résultats
    """
    lib = get_lib()
    result = lib.benchmark_compare(n_points, n_coefficients, n_iterations)
    
    return {
        'n_points': result.n_points,
        'n_coefficients': result.n_coefficients,
        'time_naive_ms': result.time_naive_ms,
        'time_openblas_ms': result.time_openblas_ms,
        'speedup': result.speedup
    }


def print_config():
    """Affiche la configuration OpenBLAS."""
    lib = get_lib()
    lib.print_openblas_config()


if __name__ == "__main__":
    print("=== Test du wrapper Python ===")
    
    # Créer un cercle
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Tester les deux implémentations
    for use_openblas in [False, True]:
        impl = "OpenBLAS" if use_openblas else "Naïve"
        wrapper = FourierWrapper(use_openblas=use_openblas)
        
        descriptors = wrapper.compute_descriptors(circle, n_coefficients=32)
        print(f"\n{impl}: {len(descriptors)} descripteurs")
        print(f"  Premiers: {descriptors[:5]}")
