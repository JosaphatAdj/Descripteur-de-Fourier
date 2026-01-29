"""
Visualisation des résultats
============================

Ce module génère des graphiques pour le rapport et la présentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Dossier de sortie
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "figures"


def setup_output_dir():
    """Crée le dossier de sortie si nécessaire."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_benchmark_comparison(results: list, save=True):
    """
    Génère le graphique comparatif naïf vs OpenBLAS.
    
    Args:
        results: Liste de dict avec 'n_points', 'time_naive_ms', 
                 'time_openblas_ms', 'speedup'
    """
    setup_output_dir()
    
    n_points = [r['n_points'] for r in results]
    time_naive = [r['time_naive_ms'] for r in results]
    time_openblas = [r['time_openblas_ms'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Temps de calcul
    ax1 = axes[0]
    width = 0.35
    x = np.arange(len(n_points))
    
    bars1 = ax1.bar(x - width/2, time_naive, width, label='Naïf', color='#e74c3c')
    bars2 = ax1.bar(x + width/2, time_openblas, width, label='OpenBLAS', color='#27ae60')
    
    ax1.set_xlabel('Nombre de points', fontsize=12)
    ax1.set_ylabel('Temps (ms)', fontsize=12)
    ax1.set_title('Comparaison des temps de calcul', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_points)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Graphique 2: Speedup
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(speedups)))
    bars = ax2.bar(x, speedups, color=colors, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
    ax2.set_xlabel('Nombre de points', fontsize=12)
    ax2.set_ylabel('Accélération (×)', fontsize=12)
    ax2.set_title('Facteur d\'accélération OpenBLAS', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_points)
    
    # Ajouter les valeurs sur les barres
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.annotate(f'{speedup:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / 'benchmark_comparison.pdf', bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {OUTPUT_DIR / 'benchmark_comparison.png'}")
    
    return fig


def plot_blas_operations(results: dict, save=True):
    """
    Génère le graphique pour les opérations BLAS Level 1.
    
    Args:
        results: Dict avec les temps pour ddot et dnrm2
    """
    setup_output_dir()
    
    operations = ['ddot\n(produit scalaire)', 'dnrm2\n(norme)']
    naive_times = [results['ddot_naive'], results['dnrm2_naive']]
    openblas_times = [results['ddot_openblas'], results['dnrm2_openblas']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(operations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, naive_times, width, label='Naïf', color='#e74c3c')
    bars2 = ax.bar(x + width/2, openblas_times, width, label='OpenBLAS (BLAS L1)', color='#3498db')
    
    ax.set_xlabel('Opération', fontsize=12)
    ax.set_ylabel('Temps (ms)', fontsize=12)
    ax.set_title('Performance des opérations BLAS Level 1\n(vecteur de 10000 éléments, 10000 itérations)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'blas_operations.png', dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {OUTPUT_DIR / 'blas_operations.png'}")
    
    return fig


def plot_contour_and_descriptors(contour: np.ndarray, descriptors: np.ndarray, 
                                  name: str, save=True):
    """
    Génère un graphique montrant le contour et ses descripteurs.
    """
    setup_output_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Contour
    ax1 = axes[0]
    ax1.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)
    ax1.plot(contour[0, 0], contour[0, 1], 'go', markersize=10, label='Début')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Contour: {name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Descripteurs
    ax2 = axes[1]
    ax2.bar(range(len(descriptors)), descriptors, color='#9b59b6', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Indice du descripteur')
    ax2.set_ylabel('Valeur normalisée')
    ax2.set_title('Descripteurs de Fourier', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f'contour_{name.lower().replace(" ", "_")}.png'
        plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {OUTPUT_DIR / filename}")
    
    return fig


def plot_scalability(results: list, save=True):
    """
    Génère le graphique de scalabilité.
    """
    setup_output_dir()
    
    n_points = [r['n_points'] for r in results]
    time_naive = [r['time_naive_ms'] for r in results]
    time_openblas = [r['time_openblas_ms'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_points, time_naive, 'o-', label='Naïf', color='#e74c3c', 
            linewidth=2, markersize=8)
    ax.plot(n_points, time_openblas, 's-', label='OpenBLAS', color='#27ae60', 
            linewidth=2, markersize=8)
    
    ax.set_xlabel('Nombre de points du contour', fontsize=12)
    ax.set_ylabel('Temps de calcul (ms)', fontsize=12)
    ax.set_title('Scalabilité: Temps vs Taille du problème', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Ajouter une ligne de tendance O(n²) pour le naïf
    x_fit = np.array(n_points)
    y_fit = time_naive[0] * (x_fit / n_points[0]) ** 2
    ax.plot(x_fit, y_fit, '--', color='#e74c3c', alpha=0.5, label='O(n²) théorique')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'scalability.png', dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {OUTPUT_DIR / 'scalability.png'}")
    
    return fig


def generate_all_figures():
    """Génère toutes les figures pour le rapport."""
    print("=== Génération des figures ===\n")
    
    # Données simulées pour démonstration
    # (À remplacer par les vrais résultats du benchmark)
    
    benchmark_results = [
        {'n_points': 100, 'time_naive_ms': 0.5, 'time_openblas_ms': 0.3, 'speedup': 1.7},
        {'n_points': 500, 'time_naive_ms': 5.0, 'time_openblas_ms': 1.2, 'speedup': 4.2},
        {'n_points': 1000, 'time_naive_ms': 18.0, 'time_openblas_ms': 2.5, 'speedup': 7.2},
        {'n_points': 2000, 'time_naive_ms': 70.0, 'time_openblas_ms': 5.0, 'speedup': 14.0},
        {'n_points': 5000, 'time_naive_ms': 450.0, 'time_openblas_ms': 12.0, 'speedup': 37.5},
    ]
    
    blas_results = {
        'ddot_naive': 15.0,
        'ddot_openblas': 2.0,
        'dnrm2_naive': 12.0,
        'dnrm2_openblas': 1.5,
    }
    
    # Générer les graphiques
    plot_benchmark_comparison(benchmark_results)
    plot_blas_operations(blas_results)
    plot_scalability(benchmark_results)
    
    # Contours de démonstration
    n = 200
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    descriptors_circle = np.exp(-np.arange(32) * 0.1)
    descriptors_circle[0] = 1.0
    plot_contour_and_descriptors(circle, descriptors_circle, "Cercle")
    
    print(f"\n✓ Toutes les figures ont été générées dans: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_figures()
