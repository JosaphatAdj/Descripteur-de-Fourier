# Guide de Compilation Cython

Le wrapper Fourier utilise maintenant Cython au lieu de CFFI pour un meilleur support des types complexes.

## Compilation

### Sur Linux VM (Direct)
```bash
# 1. Installer Cython
pip install cython

# 2. Compiler la bibliothèque C
cd c_src
make clean && make
cd ..

# 3. Compiler l'extension Cython
python setup.py build_ext --inplace
```

### Sur Windows avec Docker
```bash
# Le Dockerfile doit être mis à jour pour inclure la compilation Cython
make docker-build
```

L'extension compilée (`fourier_wrapper.*.so` ou `.pyd`) doit être dans le dossier `python/`.

## Utilisation

```python
from fourier_wrapper import FourierWrapper

wrapper = FourierWrapper(use_openblas=True)

# Simple
desc = wrapper.compute_descriptors(contour, n_coefficients=32)

# Batch (utilise GEMM)
descs = wrapper.compute_descriptors_batch([contour1, contour2, ...], n_coefficients=32)
```
