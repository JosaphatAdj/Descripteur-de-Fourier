"""
Script de benchmarks complet
============================

Ce script exécute les benchmarks et génère les résultats
pour le rapport.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire python au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Dossier de sortie
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def setup_dirs():
    """Crée les dossiers de sortie."""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    (RESULTS_DIR / "data").mkdir(exist_ok=True)


def run_full_benchmark():
    """Exécute le benchmark complet."""
    print("╔══════════════════════════════════════════╗")
    print("║        BENCHMARK COMPLET                 ║")
    print("║    Naïf vs OpenBLAS                      ║")
    print("╚══════════════════════════════════════════╝\n")
    
    setup_dirs()
    
    try:
        from fourier_wrapper import run_benchmark, print_config
    except ImportError as e:
        print(f"⚠ Impossible de charger la bibliothèque: {e}")
        print("→ Utilisez le mode simulation avec --simulate")
        return None
    
    # Afficher la configuration
    print("=== Configuration ===")
    print_config()
    
    # Paramètres du benchmark
    point_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    n_coefficients = 32
    n_iterations = 100
    
    results = []
    
    print(f"\n=== Benchmarks (n_coeff={n_coefficients}, iter={n_iterations}) ===")
    print(f"{'Points':<10} {'Naïf (ms)':<12} {'OpenBLAS (ms)':<14} {'Speedup':<10}")
    print("-" * 50)
    
    for n_points in point_sizes:
        result = run_benchmark(n_points, n_coefficients, n_iterations)
        results.append(result)
        
        print(f"{result['n_points']:<10} "
              f"{result['time_naive_ms']:<12.4f} "
              f"{result['time_openblas_ms']:<14.4f} "
              f"{result['speedup']:<10.2f}x")
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = RESULTS_DIR / "data" / f"benchmark_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_coefficients': n_coefficients,
            'n_iterations': n_iterations,
            'results': results
        }, f, indent=2)
    print(f"\n✓ Résultats sauvegardés: {json_path}")
    
    # CSV
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "data" / f"benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV sauvegardé: {csv_path}")
    
    # Générer les graphiques
    try:
        from visualization import (plot_benchmark_comparison, 
                                    plot_scalability, 
                                    generate_all_figures)
        
        print("\n=== Génération des graphiques ===")
        plot_benchmark_comparison(results)
        plot_scalability(results)
        
    except ImportError as e:
        print(f"⚠ Visualisation non disponible: {e}")
    
    return results


def run_simulated_benchmark():
    """Exécute un benchmark simulé (sans la bibliothèque C)."""
    print("=== Mode simulation (sans bibliothèque C) ===\n")
    
    setup_dirs()
    
    # Données simulées basées sur des résultats typiques
    point_sizes = [100, 500, 1000, 2000, 5000]
    
    results = []
    
    for n_pts in point_sizes:
        # Simulation réaliste
        # Naïf: O(n * n_coeff) avec constante élevée
        time_naive = 0.005 * n_pts + 0.00001 * n_pts * n_pts / 100
        
        # OpenBLAS: O(n log n) + overhead constant
        time_openblas = 0.1 + 0.001 * n_pts * np.log(n_pts) / 10
        
        speedup = time_naive / time_openblas
        
        results.append({
            'n_points': n_pts,
            'time_naive_ms': time_naive,
            'time_openblas_ms': time_openblas,
            'speedup': speedup,
            'n_coefficients': 32
        })
    
    print(f"{'Points':<10} {'Naïf (ms)':<12} {'OpenBLAS (ms)':<14} {'Speedup':<10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['n_points']:<10} "
              f"{r['time_naive_ms']:<12.4f} "
              f"{r['time_openblas_ms']:<14.4f} "
              f"{r['speedup']:<10.2f}x")
    
    # Générer les graphiques
    try:
        from visualization import plot_benchmark_comparison, plot_scalability
        print("\n=== Génération des graphiques ===")
        plot_benchmark_comparison(results)
        plot_scalability(results)
    except Exception as e:
        print(f"⚠ Erreur visualisation: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmarks Fourier OpenBLAS")
    parser.add_argument('--simulate', action='store_true', 
                       help="Mode simulation sans bibliothèque C")
    args = parser.parse_args()
    
    if args.simulate:
        run_simulated_benchmark()
    else:
        run_full_benchmark()
