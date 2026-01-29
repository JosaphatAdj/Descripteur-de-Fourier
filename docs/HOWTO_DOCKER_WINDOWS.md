# Exécution avec Docker depuis Windows

Ce guide explique comment compiler et exécuter le projet en utilisant Docker Desktop sur Windows. C'est la méthode recommandée car elle isole l'environnement Linux nécessaire pour OpenBLAS.

## Prérequis
*   Docker Desktop installé et lancé.

## 1. Commanes via Makefile (Recommandé)
Le fichier `Makefile` à la racine simplifie toutes les commandes Docker. Ouvrez un terminal (PowerShell ou CMD) dans le dossier du projet.

### Compiler le projet
Cette étape construit l'image Docker et compile le code C avec OpenBLAS.
```powershell
make docker-build
```

### Lancer le Benchmark "Batch Processing" (Nouveau)
Pour tester la nouvelle fonctionnalité de traitement par lot (Batch) :
```powershell
docker-compose run fourier python python/benchmarks/benchmark_batch.py
```

### Lancer l'analyse complète (Casting)
```powershell
make docker-run
```
*Note : Si vous voulez utiliser l'extraction par IA, assurez-vous que `rembg` est installé dans l'image (via le Dockerfile ou en l'installant interactivement).*

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

### Exécuter un script Python spécifique
```powershell
docker-compose run fourier python python/main.py
```

### Exécuter les tests
```powershell
docker-compose run test
```
