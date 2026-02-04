"""
Analyse des pièces de casting avec Descripteurs de Fourier
==========================================================

Ce script analyse les images de pièces métalliques (OK vs défectueuses)
en utilisant les descripteurs de Fourier calculés avec OpenBLAS.

Pipeline:
    1. Charger les images
    2. Extraire les contours
    3. Calculer les descripteurs de Fourier (C + OpenBLAS)
    4. Comparer les classes OK vs DEF
    5. Visualiser les résultats
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from contour_extraction import (
    load_dataset, 
    process_image_to_contour,
    load_image,
    binarize_image
)

# Dossiers
DATA_DIR = Path(__file__).parent.parent / "data" / "casting"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Configuration
N_POINTS = 512  # Points par contour (augmenté pour plus de précision)
N_COEFFICIENTS = 128  # Coefficients de Fourier (augmenté pour hautes fréquences)


def compute_descriptors_python(contour: np.ndarray, n_coefficients: int = 32) -> np.ndarray:
    """
    Calcule les descripteurs de Fourier en Python pur.
    (Fallback si la bibliothèque C n'est pas disponible)
    
    Args:
        contour: Contour (N, 2)
        n_coefficients: Nombre de coefficients
        
    Returns:
        Descripteurs normalisés
    """
    N = len(contour)
    
    # Représentation complexe
    z = contour[:, 0] + 1j * contour[:, 1]
    
    # FFT
    coefficients = np.fft.fft(z)
    coefficients = np.fft.fftshift(coefficients) / N
    
    # Garder les coefficients centraux
    center = N // 2
    half_n = n_coefficients // 2
    coeffs = coefficients[center - half_n : center + half_n + 1]
    
    # Normaliser (Normalisation L1: somme des magnitudes = 1)
    # Plus robuste que |c_1| pour des formes irrégulières
    magnitudes = np.abs(coeffs)
    
    # Exclure c_0 (centre)
    center_idx = len(coeffs) // 2
    mask = np.ones(len(coeffs), dtype=bool)
    mask[center_idx] = False
    
    descriptors = magnitudes[mask]
    
    # Diviser par la somme (L1) ou max (Linf)
    total_magnitude = np.sum(descriptors)
    
    if total_magnitude > 1e-10:
        descriptors = descriptors / total_magnitude
    
    return descriptors


def try_load_c_library():
    """Tente de charger la bibliothèque C."""
    try:
        from fourier_wrapper import FourierWrapper
        wrapper = FourierWrapper(use_openblas=True)
        # Test rapide
        test_contour = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 100)),
                                        np.sin(np.linspace(0, 2*np.pi, 100))])
        _ = wrapper.compute_descriptors(test_contour, 16)
        return wrapper
    except Exception as e:
        print(f"⚠ Bibliothèque C non disponible: {e}")
        print("→ Utilisation du fallback Python")
        return None


def analyze_casting_dataset():
    """
    Analyse complète du dataset de casting.
    
    Returns:
        Dictionnaire avec les résultats
    """
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Analyse des Pièces de Casting                       ║")
    print("║  Descripteurs de Fourier (Naïf vs OpenBLAS)          ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    
    # Créer le dossier de résultats
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Charger la bibliothèque C si disponible
    c_wrapper = try_load_c_library()
    use_c_lib = c_wrapper is not None
    
    # Charger le dataset
    print("\n=== Chargement du dataset ===")
    contours_ok, contours_def, files_ok, files_def = load_dataset(str(DATA_DIR))
    
    if not contours_ok or not contours_def:
        print("❌ Erreur: Aucun contour extrait")
        return None
    
    # Calculer les descripteurs
    print("\n=== Calcul des descripteurs de Fourier ===")
    
    descriptors_ok = []
    descriptors_def = []
    
    # Mesurer le temps
    start_time = time.perf_counter()
    
    for contour in contours_ok:
        if use_c_lib:
            desc = c_wrapper.compute_descriptors(contour, N_COEFFICIENTS)
        else:
            desc = compute_descriptors_python(contour, N_COEFFICIENTS)
        descriptors_ok.append(desc)
    
    for contour in contours_def:
        if use_c_lib:
            desc = c_wrapper.compute_descriptors(contour, N_COEFFICIENTS)
        else:
            desc = compute_descriptors_python(contour, N_COEFFICIENTS)
        descriptors_def.append(desc)
    
    elapsed = time.perf_counter() - start_time
    
    method = "OpenBLAS" if use_c_lib else "Python"
    print(f"✓ Méthode: {method}")
    print(f"✓ Temps total: {elapsed*1000:.2f} ms")
    print(f"✓ Temps/image: {elapsed*1000/(len(contours_ok)+len(contours_def)):.2f} ms")
    
    # Convertir en arrays
    descriptors_ok = np.array(descriptors_ok)
    descriptors_def = np.array(descriptors_def)
    
    print(f"\n✓ Shape OK: {descriptors_ok.shape}")
    print(f"✓ Shape DEF: {descriptors_def.shape}")
    
    # Analyser les différences
    print("\n=== Analyse des descripteurs ===")
    
    # Moyenne par classe
    mean_ok = descriptors_ok.mean(axis=0)
    mean_def = descriptors_def.mean(axis=0)
    std_ok = descriptors_ok.std(axis=0)
    std_def = descriptors_def.std(axis=0)
    
    # Distance moyenne entre classes
    from scipy.spatial.distance import cdist
    cross_distances = cdist(descriptors_ok, descriptors_def)
    intra_ok = cdist(descriptors_ok, descriptors_ok)
    intra_def = cdist(descriptors_def, descriptors_def)
    
    # Masquer la diagonale pour intra-classe
    np.fill_diagonal(intra_ok, np.nan)
    np.fill_diagonal(intra_def, np.nan)
    
    print(f"\nDistances (Euclidienne):")
    print(f"  Intra-classe OK:  {np.nanmean(intra_ok):.4f} ± {np.nanstd(intra_ok):.4f}")
    print(f"  Intra-classe DEF: {np.nanmean(intra_def):.4f} ± {np.nanstd(intra_def):.4f}")
    print(f"  Inter-classe:     {cross_distances.mean():.4f} ± {cross_distances.std():.4f}")
    
    # Ratio de séparabilité
    separability = cross_distances.mean() / (np.nanmean(intra_ok) + np.nanmean(intra_def) + 1e-10)
    print(f"\n  Ratio de séparabilité (Global): {separability:.2f}")
    
    # --- Analyse Hautes Fréquences ---
    # Ignorer les basses fréquences (forme globale) et se concentrer sur les détails
    cutoff = 10
    if descriptors_ok.shape[1] > cutoff:
        ok_hf = descriptors_ok[:, cutoff:]
        def_hf = descriptors_def[:, cutoff:]
        
        # Recalculer les distances sur HF
        cross_hf = cdist(ok_hf, def_hf)
        intra_ok_hf = cdist(ok_hf, ok_hf); np.fill_diagonal(intra_ok_hf, np.nan)
        intra_def_hf = cdist(def_hf, def_hf); np.fill_diagonal(intra_def_hf, np.nan)
        
        sep_hf = cross_hf.mean() / (np.nanmean(intra_ok_hf) + np.nanmean(intra_def_hf) + 1e-10)
        print(f"  Ratio de séparabilité (Hautes Fréquences > {cutoff}): {sep_hf:.2f}")
        
        # Metrique simple: Somme des énergies HF
        energy_ok = np.sum(ok_hf, axis=1)
        energy_def = np.sum(def_hf, axis=1)
        print(f"\n  Énergie HF moyenne OK:  {energy_ok.mean():.4f} ± {energy_ok.std():.4f}")
        print(f"  Énergie HF moyenne DEF: {energy_def.mean():.4f} ± {energy_def.std():.4f}")
    
    if separability > 1.0:
        print("  ✓ Bonne séparation entre les classes!")
    else:
        print("  ⚠ Séparation faible, les classes se chevauchent")
    
    # Résultats
    results = {
        'descriptors_ok': descriptors_ok,
        'descriptors_def': descriptors_def,
        'files_ok': files_ok,
        'files_def': files_def,
        'mean_ok': mean_ok,
        'mean_def': mean_def,
        'std_ok': std_ok,
        'std_def': std_def,
        'cross_distances': cross_distances,
        'separability': separability,
        'method': method,
        'time_ms': elapsed * 1000
    }
    
    return results


def visualize_results(results: Dict):
    """Génère les visualisations."""
    
    print("\n=== Génération des visualisations ===")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =========================================
    # Figure 1: Comparaison des descripteurs
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Moyennes des descripteurs par classe
    ax1 = axes[0, 0]
    x = np.arange(len(results['mean_ok']))
    width = 0.35
    
    ax1.bar(x - width/2, results['mean_ok'], width, label='OK', color='#27ae60', alpha=0.8)
    ax1.bar(x + width/2, results['mean_def'], width, label='Défectueux', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Indice du descripteur')
    ax1.set_ylabel('Valeur moyenne')
    ax1.set_title('Descripteurs de Fourier moyens par classe', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. Écart-type par classe
    ax2 = axes[0, 1]
    ax2.fill_between(x, results['mean_ok'] - results['std_ok'], 
                     results['mean_ok'] + results['std_ok'], alpha=0.3, color='#27ae60')
    ax2.fill_between(x, results['mean_def'] - results['std_def'],
                     results['mean_def'] + results['std_def'], alpha=0.3, color='#e74c3c')
    ax2.plot(x, results['mean_ok'], 'o-', color='#27ae60', label='OK')
    ax2.plot(x, results['mean_def'], 's-', color='#e74c3c', label='Défectueux')
    ax2.set_xlabel('Indice du descripteur')
    ax2.set_ylabel('Valeur (moyenne ± std)')
    ax2.set_title('Distribution des descripteurs', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Différence absolue
    ax3 = axes[1, 0]
    diff = np.abs(results['mean_ok'] - results['mean_def'])
    colors = plt.cm.Reds(diff / diff.max())
    bars = ax3.bar(x, diff, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Indice du descripteur')
    ax3.set_ylabel('|OK - DEF|')
    ax3.set_title('Différence absolue entre classes', fontsize=12, fontweight='bold')
    
    # Marquer les descripteurs les plus discriminants
    top_indices = np.argsort(diff)[-5:]
    for idx in top_indices:
        ax3.annotate(f'{idx}', (idx, diff[idx]), ha='center', va='bottom', fontweight='bold')
    
    # 4. Matrice de distances
    ax4 = axes[1, 1]
    all_desc = np.vstack([results['descriptors_ok'], results['descriptors_def']])
    n_ok = len(results['descriptors_ok'])
    
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(all_desc))
    
    im = ax4.imshow(dist_matrix, cmap='viridis')
    ax4.axhline(n_ok - 0.5, color='white', linewidth=2, linestyle='--')
    ax4.axvline(n_ok - 0.5, color='white', linewidth=2, linestyle='--')
    ax4.set_title('Matrice de distances', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Échantillon')
    ax4.set_ylabel('Échantillon')
    plt.colorbar(im, ax=ax4, label='Distance')
    
    # Labels
    ax4.text(n_ok/2, -1, 'OK', ha='center', fontweight='bold', color='#27ae60')
    ax4.text(n_ok + len(results['descriptors_def'])/2, -1, 'DEF', ha='center', fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'casting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé: {RESULTS_DIR / 'casting_analysis.png'}")
    
    # =========================================
    # Figure 2: PCA 2D
    # =========================================
    from sklearn.decomposition import PCA
    
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    pca = PCA(n_components=2)
    all_desc_pca = pca.fit_transform(all_desc)
    
    ok_pca = all_desc_pca[:n_ok]
    def_pca = all_desc_pca[n_ok:]
    
    ax.scatter(ok_pca[:, 0], ok_pca[:, 1], c='#27ae60', s=150, 
               label='OK', edgecolor='black', linewidth=1, alpha=0.8, marker='o')
    ax.scatter(def_pca[:, 0], def_pca[:, 1], c='#e74c3c', s=150, 
               label='Défectueux', edgecolor='black', linewidth=1, alpha=0.8, marker='X')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Projection PCA des Descripteurs de Fourier\nPièces OK vs Défectueuses', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / 'casting_pca.png', dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé: {RESULTS_DIR / 'casting_pca.png'}")

    # =========================================
    # Figure 3: Analyse détaillée de 4 cas
    # =========================================
    print("\n=== Génération visualisation détaillée (4 cas) ===")
    
    # Sélectionner 2 OK et 2 DEF
    indices_ok = [0, 1] if len(results['files_ok']) >= 2 else range(len(results['files_ok']))
    indices_def = [0, 1] if len(results['files_def']) >= 2 else range(len(results['files_def']))
    
    samples = []
    for i in indices_ok:
        samples.append({
            'type': 'OK',
            'file': results['files_ok'][i],
            'desc': results['descriptors_ok'][i],
            'path': DATA_DIR / 'ok' / results['files_ok'][i]
        })
    for i in indices_def:
        samples.append({
            'type': 'DEF',
            'file': results['files_def'][i],
            'desc': results['descriptors_def'][i],
            'path': DATA_DIR / 'def' / results['files_def'][i]
        })
        
    fig3, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    for i, sample in enumerate(samples):
        # 1. Image originale
        img = load_image(str(sample['path']))
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"{sample['type']} - {sample['file']}", fontweight='bold')
        axes[i, 0].axis('off')
        
        # 2. Contour extrait
        contour = process_image_to_contour(str(sample['path']), n_points=N_POINTS)
        
        # Debug info
        print(f"  Sample {i}: Contour Range X[{contour[:,0].min():.2f}, {contour[:,0].max():.2f}] Y[{contour[:,1].min():.2f}, {contour[:,1].max():.2f}]")
        
        axes[i, 1].plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)
        axes[i, 1].set_aspect('equal')
        # Centrer sur 0,0 et fixer l'échelle
        axes[i, 1].set_xlim(-1.2, 1.2)
        axes[i, 1].set_ylim(-1.2, 1.2)
        axes[i, 1].axhline(0, color='gray', linestyle='dotted')
        axes[i, 1].axvline(0, color='gray', linestyle='dotted')
        axes[i, 1].set_title(f"Contour extrait ({len(contour)} pts)")
        
        # 3. Descripteurs (Spectre)
        desc = sample['desc']
        print(f"  Sample {i}: Desc Max={desc.max():.4f}, Sum={desc.sum():.4f}")
        
        x = range(len(desc))
        axes[i, 2].bar(x, desc, color='#27ae60' if sample['type']=='OK' else '#e74c3c')
        axes[i, 2].set_xlabel('Harmonique')
        axes[i, 2].set_ylabel('Amplitude normalisée')
        # Echelle Log pour mieux voir les petites valeurs HF
        # axes[i, 2].set_yscale('log') 
        axes[i, 2].set_ylim(0, max(0.1, desc.max() * 1.1)) # Minimum 0.1 pour voir l'axe
        axes[i, 2].set_title(f"Signature spectrale (Max={desc.max():.2f})")
        axes[i, 2].grid(True, alpha=0.3)

        
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / 'casting_detailed_samples.png', dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé: {RESULTS_DIR / 'casting_detailed_samples.png'}")

    plt.show()


def main():
    """Point d'entrée principal."""
    
    # Vérifier que les données existent
    if not DATA_DIR.exists():
        print(f"❌ Dossier non trouvé: {DATA_DIR}")
        return
    
    # Analyser
    results = analyze_casting_dataset()
    
    if results is None:
        return
    
    # Visualiser
    visualize_results(results)
    
    print("\n" + "="*50)
    print("✓ ANALYSE TERMINÉE")
    print("="*50)


if __name__ == "__main__":
    main()
