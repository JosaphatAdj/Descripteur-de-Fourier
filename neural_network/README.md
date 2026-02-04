# Neural Network Dense avec OpenBLAS

Implémentation from scratch d'un réseau de neurones dense en **C pur + OpenBLAS**.

## Architecture

```
Input:  784 (28×28 pixels MNIST)
Hidden: 128 neurons + ReLU
Output: 10 classes + Softmax
```

## Utilisation d'OpenBLAS

**100% des opérations matricielles = GEMM !**

| Opération | Routine OpenBLAS | Contribution |
|-----------|------------------|--------------|
| X·W1 | `cblas_dgemm` | 35% |
| A1·W2 | `cblas_dgemm` | 15% |
| Y1ᵀ·dY2 | `cblas_dgemm` | 20% |
| dY2·W2ᵀ | `cblas_dgemm` | 15% |
| Xᵀ·dY1 | `cblas_dgemm` | 15% |
| **Total GEMM** | | **100%** |

→ Performance maximale attendue : **~5-6× plus rapide** que NumPy

## Installation

### 1. Compiler la bibliothèque C

```bash
cd neural_network/c_src
make
```

Cela génère `lib/libnn.so`.

### 2. Télécharger MNIST

```bash
cd neural_network
python download_mnist.py
```

### 3. Entraîner

```bash
python train.py
```

## Performance Attendue

- **C + OpenBLAS** : ~2-3s par epoch (60k images)
- **Python/NumPy** : ~15s par epoch
- **Speedup** : **~5-6×**

## Fichiers

```
neural_network/
├── c_src/
│   ├── include/        # Headers (nn.h, matrix.h, mnist.h)
│   ├── matrix.c        # Wrappers OpenBLAS
│   ├── nn.c            # Forward/Backward (5 GEMM)
│   ├── mnist.c         # Loader dataset
│   └── Makefile
├── train.py            # Script d'entraînement
├── download_mnist.py   # Téléchargement MNIST
└── data/mnist/         # Dataset (téléchargé)
```

## Résultats Attendus

Après 10 epochs :
- **Accuracy** : >95% sur MNIST
- **Loss** : <0.1
