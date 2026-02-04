# Exécution avec Docker depuis Windows

Ce guide explique comment compiler et exécuter le projet en utilisant Docker Desktop sur Windows. C'est la méthode recommandée car elle isole l'environnement Linux nécessaire pour OpenBLAS et Cython.

## Prérequis
*   Docker Desktop installé et lancé.

## 1. Commandes via Makefile (Recommandé)
Le fichier `Makefile` à la racine simplifie toutes les commandes Docker. Ouvrez un terminal (PowerShell ou CMD) dans le dossier du projet.

### Compiler le projet (C + Cython)
Cette étape construit l'image Docker, compile le code C avec OpenBLAS, puis compile l'extension Cython.
```powershell
make docker-build
```

**Note :** Le Dockerfile exécute automatiquement :
1. Compilation de la bibliothèque C (`libfourier.so`)
2. Installation de Cython
3. Compilation de l'extension Cython (`fourier_wrapper.*.so`)

### Lancer l'analyse complète (Casting)
```powershell
make docker-run
```
*Utilise le wrapper Cython avec support complet des types complexes.*

### Lancer le Benchmark "Batch Processing" (GEMM)
Pour tester le batch processing avec BLAS Level 3 :
```powershell
docker-compose run fourier python python/benchmarks/benchmark_batch.py
```

### Accéder au Shell (Terminal Linux)
Si vous voulez explorer ou lancer des commandes manuelles à l'intérieur du conteneur :
```powershell
make docker-shell
```

---

## 2. Commandes Docker Manuelles (Alternative)
Si vous n'avez pas `make` installé sur Windows, utilisez directement `docker-compose`.

### Construire
```powershell
docker-compose build
```

### Exécuter l'analyse
```powershell
docker-compose run fourier python python/analyze_casting.py
```

### Exécuter les benchmarks
```powershell
docker-compose run fourier python python/benchmarks/benchmark_batch.py
```

