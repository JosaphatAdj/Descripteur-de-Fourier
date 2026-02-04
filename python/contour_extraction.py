"""
Extraction de contours à partir d'images
=========================================

Ce module fournit des fonctions pour charger des images,
les binariser, et extraire les contours pour l'analyse.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """
    Charge une image depuis un fichier.
    
    Args:
        path: Chemin vers l'image
        grayscale: Si True, charge en niveaux de gris
        
    Returns:
        Image sous forme d'array numpy
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    
    if img is None:
        raise FileNotFoundError(f"Impossible de charger: {path}")
    
    return img


def binarize_image(img: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    Binarise une image en niveaux de gris.
    ADAPTÉ pour des contours fins (LLM-generated) sur fond sombre.
    
    Args:
        img: Image en niveaux de gris
        method: 'otsu', 'adaptive', ou 'canny'
        
    Returns:
        Image binaire (0 ou 255)
    """
    # Détecter si c'est déjà une image de contour (très peu de pixels non-noirs)
    white_ratio = np.mean(img > 20) / np.prod(img.shape)
    print(f"  DEBUG: Ratio pixels non-noirs = {white_ratio:.3f}")
    
    if white_ratio < 0.05:  # Moins de 5% de pixels non-noirs = contour fin
        print("  DEBUG: Image détectée comme contour fin, traitement minimal")
        # Seuillage simple sans morphologie agressive
        _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        
        # Très légère dilatation pour connecter les gaps (1px)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    # Cas standard (image avec texture)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    if method == 'otsu':
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'canny':
        high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        binary = cv2.Canny(blurred, low_thresh, high_thresh)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    # Opérations morphologiques DOUCES (kernel réduit)
    if method != 'canny':
        kernel = np.ones((3,3), np.uint8)  # Réduit de 5x5 à 3x3
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Inversion automatique
    h, w = binary.shape
    mask_border = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_border, (0, 0), (w-1, h-1), 255, thickness=5)
    
    mean_border = cv2.mean(binary, mask=mask_border)[0]
    
    if mean_border > 127:
        binary = cv2.bitwise_not(binary)
    
    return binary


def remove_background_rembg(img: np.ndarray) -> np.ndarray:
    """
    Supprime l'arrière-plan avec rembg (U-2-Net).
    Retourne un masque binaire propre de la pièce.
    """
    from rembg import remove
    
    # rembg attend RGB ou RGBA
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
    
    # Enlever le fond (retourne RGBA)
    result = remove(img_rgb)
    
    # Extraire le canal Alpha comme masque
    if result.shape[2] == 4:
        alpha = result[:, :, 3]
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        return binary
    
    # Fallback si pas d'alpha
    return cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)


def extract_contour(binary_img: np.ndarray, 
                    largest_only: bool = True) -> Optional[np.ndarray]:
    """
    Extrait le contour d'une image binaire.
    
    Args:
        binary_img: Image binaire
        largest_only: Si True, retourne uniquement le plus grand contour
        
    Returns:
        Contour sous forme d'array (N, 2) ou None si aucun contour
    """
    # Trouver les contours (RETR_EXTERNAL pour ignorer les trous intérieurs)
    contours, _ = cv2.findContours(
        binary_img, 
        cv2.RETR_EXTERNAL,  # Seulement le contour externe !
        cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        print(f"  DEBUG: Aucun contour trouvé dans l'image binaire")
        return None
    
    print(f"  DEBUG: {len(contours)} contour(s) trouvé(s)")
    
    if largest_only:
        # Prendre le plus grand contour (la pièce)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        print(f"  DEBUG: Plus grand contour - Area={area:.0f}, Shape={largest.shape}")
        
        # Convertir de (N, 1, 2) à (N, 2)
        result = largest.squeeze()
        print(f"  DEBUG: Après squeeze - Shape={result.shape}")
        
        # Vérifier si le squeeze a trop réduit (cas pathologique: contour à 1 point)
        if result.ndim == 1:
            print(f"  WARNING: Contour dégénéré (1 seul point ou dimension incorrecte)")
            return None
            
        return result
    else:
        return [c.squeeze() for c in contours]


def resample_contour(contour: np.ndarray, n_points: int = 256) -> np.ndarray:
    """
    Rééchantillonne un contour pour avoir un nombre fixe de points.
    
    Args:
        contour: Contour original (N, 2)
        n_points: Nombre de points souhaité
        
    Returns:
        Contour rééchantillonné (n_points, 2)
    """
    # Calculer la longueur cumulée
    diff = np.diff(contour, axis=0)
    distances = np.sqrt((diff ** 2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]
    
    # Points équidistants
    target_positions = np.linspace(0, total_length, n_points, endpoint=False)
    
    # Interpoler
    resampled = np.zeros((n_points, 2))
    for i, pos in enumerate(target_positions):
        idx = np.searchsorted(cumulative, pos) - 1
        idx = max(0, min(idx, len(contour) - 2))
        
        if cumulative[idx + 1] - cumulative[idx] > 0:
            t = (pos - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx])
        else:
            t = 0
        
        resampled[i] = (1 - t) * contour[idx] + t * contour[idx + 1]
    
    return resampled


def normalize_contour(contour: np.ndarray) -> np.ndarray:
    """
    Normalise un contour pour l'invariance:
    - Translation: centrer sur l'origine
    - Échelle: normaliser par la distance max au centre
    
    Args:
        contour: Contour (N, 2)
        
    Returns:
        Contour normalisé
    """
    # Centrer
    centroid = contour.mean(axis=0)
    centered = contour - centroid
    
    # Normaliser l'échelle
    max_dist = np.sqrt((centered ** 2).sum(axis=1)).max()
    if max_dist > 0:
        normalized = centered / max_dist
    else:
        normalized = centered
    
    return normalized


def process_image_to_contour(image_path: str, 
                              n_points: int = 256,
                              normalize: bool = True) -> np.ndarray:
    """
    Pipeline complet : image -> contour normalisé.
    
    Args:
        image_path: Chemin vers l'image
        n_points: Nombre de points du contour
        normalize: Si True, normalise le contour
        
    Returns:
        Contour prêt pour l'analyse (n_points, 2)
    """
    # Charger
    img = load_image(image_path)
    
    # Supprimer l'arrière-plan avec IA (rembg)
    try:
        from contour_extraction import remove_background_rembg
        binary = remove_background_rembg(img)
    except ImportError:
        print("⚠ rembg non installé, fallback sur Otsu")
        binary = binarize_image(img, method='otsu')
    except Exception as e:
        print(f"⚠ Erreur rembg: {e}, fallback sur Otsu")
        binary = binarize_image(img, method='otsu')
    
    # Extraire le contour
    contour = extract_contour(binary)

    
    if contour is None or len(contour) < 10:
        raise ValueError(f"Contour invalide pour {image_path}")
    
    # Rééchantillonner
    resampled = resample_contour(contour, n_points)
    
    # Normaliser
    if normalize:
        resampled = normalize_contour(resampled)
    
    return resampled


def load_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                          List[str], List[str]]:
    """
    Charge le dataset de casting.
    
    Args:
        data_dir: Chemin vers data/casting/ ou data/originaux/
        
    Returns:
        (contours_ok, contours_def, files_ok, files_def)
    """
    data_path = Path(data_dir)
    
    ok_dir = data_path / "ok"
    def_dir = data_path / "def"
    
    contours_ok = []
    contours_def = []
    files_ok = []
    files_def = []
    
    # Extensions d'image supportées
    image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.bmp', '*.tif', '*.tiff']
    
    # Charger les pièces OK
    if ok_dir.exists():
        for ext in image_extensions:
            for img_path in sorted(ok_dir.glob(ext)):
                try:
                    contour = process_image_to_contour(str(img_path))
                    contours_ok.append(contour)
                    files_ok.append(img_path.name)
                except Exception as e:
                    print(f"⚠ Erreur {img_path.name}: {e}")
    
    # Charger les pièces défectueuses
    if def_dir.exists():
        for ext in image_extensions:
            for img_path in sorted(def_dir.glob(ext)):
                try:
                    contour = process_image_to_contour(str(img_path))
                    contours_def.append(contour)
                    files_def.append(img_path.name)
                except Exception as e:
                    print(f"⚠ Erreur {img_path.name}: {e}")
    
    print(f"✓ Chargé: {len(contours_ok)} OK, {len(contours_def)} DEF")
    
    return contours_ok, contours_def, files_ok, files_def


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test sur une image
    DATA_DIR = Path(__file__).parent.parent / "data" / "casting"
    
    # Charger une image OK et une DEF
    ok_path = DATA_DIR / "ok_front" / "cast_ok_0_119.jpeg"
    def_path = DATA_DIR / "def_front" / "cast_def_0_0.jpeg"
    
    print("=== Test extraction de contours ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (path, label) in enumerate([(ok_path, "OK"), (def_path, "DEF")]):
        # Image originale
        img = load_image(str(path))
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"{label} - Original")
        
        # Image binaire
        binary = binarize_image(img)
        axes[i, 1].imshow(binary, cmap='gray')
        axes[i, 1].set_title(f"{label} - Binaire")
        
        # Contour
        contour = process_image_to_contour(str(path), n_points=256)
        axes[i, 2].plot(contour[:, 0], contour[:, 1], 'b-', linewidth=1)
        axes[i, 2].set_aspect('equal')
        axes[i, 2].set_title(f"{label} - Contour ({len(contour)} points)")
    
    plt.tight_layout()
    plt.savefig(DATA_DIR.parent.parent / "results" / "contour_extraction.png", dpi=150)
    print("✓ Figure sauvegardée: results/contour_extraction.png")
    plt.show()
