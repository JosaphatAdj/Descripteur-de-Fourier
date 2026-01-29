# Optimisation des Descripteurs de Fourier avec OpenBLAS

## 📋 Description

Ce projet implémente le calcul des descripteurs de Fourier pour l'analyse de formes avec deux approches :
- **Naïve** : Boucles C pures (baseline)
- **OpenBLAS** : Utilisation des fonctions BLAS Level 1 (`ddot`, `dnrm2`)

L'objectif est de démontrer l'accélération obtenue grâce à OpenBLAS.

## 🗂 Structure du Projet

```
openblas/
├── c_src/                      # Code source C
│   ├── include/
│   │   └── fourier.h           # API publique
│   ├── fourier_naive.c         # Implémentation naïve
│   ├── fourier_openblas.c      # Implémentation OpenBLAS
│   ├── contour.c               # Gestion des contours
│   ├── utils.c                 # Utilitaires et benchmarks
│   ├── test_main.c             # Programme de test
│   └── Makefile                # Compilation C
├── python/                     # Wrappers et visualisation
│   ├── fourier_wrapper.py      # Interface CFFI
│   ├── main.py                 # Point d'entrée
│   ├── visualization.py        # Génération des graphiques
│   ├── benchmarks/
│   │   └── run_benchmarks.py   # Script de benchmarks
│   └── lib/                    # Bibliothèque .so compilée
├── data/                       # Données de test
├── results/                    # Résultats et figures
├── about/
│   └── overview                # Plan du rapport
├── Dockerfile                  # Image Docker
├── docker-compose.yml          # Services Docker
├── Makefile                    # Commandes principales
└── requirements.txt            # Dépendances Python
```

## 🚀 Utilisation

### Avec Docker (recommandé)

```bash
# Construire l'image
make docker-build

# Exécuter les tests C + Python
make docker-test

# Ouvrir un shell interactif
make docker-shell

# Lancer les benchmarks
make docker-benchmark
```

### Dans le container Docker

```bash
# Compiler le code C
cd c_src && make all

# Exécuter le test C
./test_fourier

# Exécuter Python
python python/main.py
```

## 📊 Fonctions BLAS utilisées

| Fonction | Description | Implémentation |
|----------|-------------|----------------|
| `cblas_ddot` | Produit scalaire | `dot_product_openblas()` |
| `cblas_dnrm2` | Norme euclidienne | `norm_openblas()` |
| `cblas_daxpy` | y = αx + y | `distance_openblas()` |
| `cblas_zdotu` | Produit scalaire complexe | `fourier_coefficients_openblas()` |

## 📈 Résultats attendus

- **Accélération** : 5-50× selon la taille du problème
- **Scalabilité** : Meilleure performance pour grands vecteurs
- **Précision** : Résultats identiques entre naïf et OpenBLAS

## 📖 Rapport

Le rapport se trouve dans `docs/rapport/` et couvre :
1. Fondements mathématiques des descripteurs de Fourier
2. Architecture d'OpenBLAS
3. Implémentation et benchmarks
4. Analyse des résultats
