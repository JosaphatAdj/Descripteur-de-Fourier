# Exécution sur Linux VM (Sans Docker)

Ce guide explique comment compiler et exécuter le projet "en direct" sur une machine virtuelle Linux (Ubuntu/Debian) ayant accès au dossier du projet (via dossier partagé).

## 1. Prérequis Système
Ouvrez un terminal dans votre VM et installez les outils de compilation et les bibliothèques OpenBLAS :

```bash
sudo apt update
sudo apt install -y build-essential gcc make libopenblas-dev liblapack-dev python3-dev python3-pip
```

## 2. Installation des Dépendances Python
Installez les paquets requis. Si vous êtes dans un environnement virtuel (recommandé) :
```bash
# Optionnel: créer venv
python3 -m venv venv
source venv/bin/activate

# Installer (inclut Cython)
pip install -r requirements.txt
```

*Note : Si vous n'utilisez pas de venv et avez une erreur "externally-managed environment", ajoutez `--break-system-packages`.*

## 3. Compilation de la Bibliothèque C
Allez dans le dossier source C et compilez :

```bash
cd /chemin/vers/le/projet/c_src
make clean
make
```

Cela doit générer le fichier `../python/lib/libfourier.so`.
Vous devriez voir : `✓ Compilation terminée`.

## 4. Compilation de l'Extension Cython
Retournez à la racine et compilez le wrapper Cython :

```bash
cd ..
python3 setup.py build_ext --inplace
```

Cela génère `fourier_wrapper.*.so` (l'extension Python compilée avec support des types complexes).

## 5. Exécution des Scripts
Lancez les scripts Python.

### Analyse Casting (Défauts)
```bash
python3 python/analyze_casting.py
```

### Benchmark Batch Processing (GEMM)
```bash
python3 python/benchmarks/benchmark_batch.py
```

### Tests Unitaires
```bash
pytest
```

---

## Troubleshooting

**Erreur "cannot find -lfourier"** : La bibliothèque C n'a pas été compilée. Retournez à l'étape 3.

**Import error sur fourier_wrapper** : L'extension Cython n'a pas été compilée ou est dans le mauvais dossier. L'exécuter depuis la racine du projet résout souvent le problème.
