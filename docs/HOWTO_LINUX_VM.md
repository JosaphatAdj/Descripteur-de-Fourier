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

# Installer
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

## 4. Exécution des Scripts
Revenez à la racine du projet et lancez les scripts Python.

### Benchmark Batch Processing (Nouveau)
```bash
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/python/lib
python3 python/benchmarks/benchmark_batch.py
```
*Note : L'export LD_LIBRARY_PATH aide parfois si le .so n'est pas trouvé, bien que le wrapper Python utilise un chemin absolu.*

### Analyse Casting (Défauts)
```bash
python3 python/analyze_casting.py
```

### Tests Unitaires
```bash
pytest
```
