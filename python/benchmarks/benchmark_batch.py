
"""
Benchmark Batch Processing
==========================

Compare les performances entre:
1. Boucle Python + OpenBLAS Level 1 (appel unitaire)
2. Batch Processing + OpenBLAS Level 3 (appel matriciel GEMM)
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le dossier parent pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_wrapper import FourierWrapper

def generate_random_contours(batch_size, n_points):
    contours = []
    t = np.linspace(0, 2*np.pi, n_points)
    for _ in range(batch_size):
        # Cercle aléatoire déformé
        r = 1.0 + 0.1 * np.random.randn(n_points)
        x = r * np.cos(t)
        y = r * np.sin(t)
        contours.append(np.column_stack([x, y]))
    return contours

def run_benchmark():
    print("=== Benchmark Batch vs Loop ===")
    
    wrapper = FourierWrapper(use_openblas=True)
    
    # Paramètres
    batch_sizes = [10, 50, 100, 500, 1000, 5000]
    n_points = 256
    n_coeffs = 32
    n_iterations = 5
    
    times_loop = []
    times_batch = []
    
    for b in batch_sizes:
        print(f"\nBatch Size: {b}")
        contours = generate_random_contours(b, n_points)
        
        # 1. Loop (Level 1)
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = [wrapper.compute_descriptors(c, n_coeffs) for c in contours]
        end = time.perf_counter()
        avg_loop = (end - start) / n_iterations
        times_loop.append(avg_loop)
        print(f"  Loop : {avg_loop*1000:.2f} ms")
        
        # 2. Batch (Level 3 GEMM)
        try:
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = wrapper.compute_descriptors_batch(contours, n_coeffs)
            end = time.perf_counter()
            avg_batch = (end - start) / n_iterations
            times_batch.append(avg_batch)
            print(f"  Batch: {avg_batch*1000:.2f} ms")
        except AttributeError:
             print("  Batch not implemented or lib not updated")
             times_batch.append(avg_loop) # Fallback pour eviter crash graph

    # Tracer
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, times_loop, 'o-', label='Boucle (BLAS Level 1)')
    plt.plot(batch_sizes, times_batch, 's-', label='Batch (BLAS Level 3 GEMM)')
    plt.xlabel('Taille du Batch (Nombre d\'images)')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Impact du Batch Processing avec OpenBLAS')
    plt.legend()
    plt.grid(True)
    
    output_path = Path(__file__).parent.parent.parent / "results" / "batch_benchmark.png"
    plt.savefig(output_path)
    print(f"\nGraphique sauvegardé: {output_path}")
    
    # Calculer speedup max
    if times_batch:
        speedup = times_loop[-1] / times_batch[-1]
        print(f"Speedup Max (N={batch_sizes[-1]}): x{speedup:.2f}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Erreur: {e}")
        print("Note: Si la lib C n'est pas recompilée avec batch support, cela échouera.")
