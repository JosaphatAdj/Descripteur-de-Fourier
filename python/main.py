"""
Script principal - Démonstration et benchmarks
===============================================
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def main():
    """Point d'entrée principal."""
    print("╔══════════════════════════════════════════╗")
    print("║  Descripteurs de Fourier avec OpenBLAS   ║")
    print("║         Démonstration Python             ║")
    print("╚══════════════════════════════════════════╝")
    
    try:
        from fourier_wrapper import FourierWrapper, run_benchmark, print_config
    except Exception as e:
        print(f"\n⚠ Erreur lors du chargement de la bibliothèque C:")
        print(f"  {e}")
        print("\n→ Assurez-vous d'avoir compilé le code C avec:")
        print("  make docker-build && docker-compose run --rm fourier make build-c")
        return
    
    # Afficher la configuration
    print("\n=== Configuration OpenBLAS ===")
    print_config()
    
    # Créer des formes de test
    print("\n=== Création des formes de test ===")
    
    n_points = 500
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    shapes = {
        'Cercle': np.column_stack([np.cos(theta), np.sin(theta)]),
        'Ellipse': np.column_stack([2*np.cos(theta), np.sin(theta)]),
        'Carré': create_square_contour(n_points),
    }
    
    for name, contour in shapes.items():
        print(f"  {name}: {len(contour)} points")
    
    # Calculer les descripteurs
    print("\n=== Calcul des descripteurs ===")
    
    wrapper_naive = FourierWrapper(use_openblas=False)
    wrapper_openblas = FourierWrapper(use_openblas=True)
    
    for name, contour in shapes.items():
        desc_naive = wrapper_naive.compute_descriptors(contour, 32)
        desc_openblas = wrapper_openblas.compute_descriptors(contour, 32)
        
        # Vérifier que les résultats sont similaires
        diff = np.max(np.abs(desc_naive - desc_openblas))
        
        print(f"\n{name}:")
        print(f"  Descripteurs: {len(desc_naive)}")
        print(f"  Diff max (naïf vs OpenBLAS): {diff:.2e}")
    
    # Benchmarks
    print("\n=== Benchmarks ===")
    print(f"{'Points':<10} {'Naïf (ms)':<12} {'OpenBLAS (ms)':<14} {'Speedup':<10}")
    print("-" * 50)
    
    for n_pts in [100, 500, 1000, 2000, 5000]:
        result = run_benchmark(n_pts, 32, 50)
        print(f"{result['n_points']:<10} "
              f"{result['time_naive_ms']:<12.4f} "
              f"{result['time_openblas_ms']:<14.4f} "
              f"{result['speedup']:<10.2f}x")
    
    print("\n✓ Démonstration terminée")


def create_square_contour(n_points: int) -> np.ndarray:
    """Crée un contour carré."""
    n_per_side = n_points // 4
    side_length = 2.0
    half = side_length / 2
    
    points = []
    
    # Bas
    for i in range(n_per_side):
        t = i / n_per_side
        points.append((-half + t * side_length, -half))
    
    # Droite
    for i in range(n_per_side):
        t = i / n_per_side
        points.append((half, -half + t * side_length))
    
    # Haut
    for i in range(n_per_side):
        t = i / n_per_side
        points.append((half - t * side_length, half))
    
    # Gauche
    for i in range(n_per_side):
        t = i / n_per_side
        points.append((-half, half - t * side_length))
    
    return np.array(points)


if __name__ == "__main__":
    main()
