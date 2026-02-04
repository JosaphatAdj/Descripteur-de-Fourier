#!/usr/bin/env python3
"""
Entraînement Neural Network avec C + OpenBLAS
==============================================

Entraîne un réseau Dense 784->128->10 sur MNIST
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

# Ajouter le dossier lib au PATH
lib_path = Path(__file__).parent / "c_src" / "lib"
sys.path.insert(0, str(lib_path))

# Charger la bibliothèque C
try:
    libnn = ctypes.CDLL(str(lib_path / "libnn.so"))
except OSError as e:
    print(f"❌ Erreur: Bibliothèque C non compilée")
    print(f"   Lancez: cd c_src && make")
    sys.exit(1)

# Définir structures C
class CMatrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int)
    ]

class CNetwork(ctypes.Structure):
    _fields_ = [
        ("W1", ctypes.POINTER(CMatrix)),
        ("b1", ctypes.POINTER(CMatrix)),
        ("W2", ctypes.POINTER(CMatrix)),
        ("b2", ctypes.POINTER(CMatrix)),
        # ... (autres champs)
    ]

class CMNISTDataset(ctypes.Structure):
    _fields_ = [
        ("images", ctypes.POINTER(CMatrix)),
        ("labels", ctypes.POINTER(ctypes.c_int)),
        ("n_samples", ctypes.c_int)
    ]

# Déclarer fonctions C
libnn.mnist_load_train.restype = ctypes.POINTER(CMNISTDataset)
libnn.mnist_load_train.argtypes = [ctypes.c_char_p]

libnn.network_create.restype = ctypes.POINTER(CNetwork)
libnn.network_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_double, ctypes.c_int]

libnn.network_train_batch.restype = ctypes.c_double
libnn.network_train_batch.argtypes = [ctypes.POINTER(CNetwork),
                                      ctypes.POINTER(CMatrix),
                                      ctypes.POINTER(CMatrix)]

libnn.network_accuracy.restype = ctypes.c_double
libnn.network_accuracy.argtypes = [ctypes.POINTER(CNetwork),
                                   ctypes.POINTER(CMatrix),
                                   ctypes.POINTER(ctypes.c_int),
                                   ctypes.c_int]

def train_mnist():
    """Entraîne le réseau sur MNIST."""
    
    print("╔══════════════════════════════════════════╗")
    print("║  Neural Network Training (C + OpenBLAS) ║")
    print("╚══════════════════════════════════════════╝\n")
    
    # Charger MNIST
    print("=== Chargement MNIST ===")
    data_dir = b"data/mnist"
    train_data = libnn.mnist_load_train(data_dir)
    
    if not train_data:
        print("❌ Erreur: MNIST non trouvé")
        print("   Lancez: python download_mnist.py")
        return
    
    n_train = train_data.contents.n_samples
    print(f"Train: {n_train} images\n")
    
    # Créer réseau
    print("=== Création du réseau ===")
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    batch_size = 64
    
    net = libnn.network_create(input_size, hidden_size, output_size,
                               learning_rate, batch_size)
    print(f"Architecture: {input_size}→{hidden_size}→{output_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}\n")
    
    # Entraînement
    print("=== Entraînement ===")
    n_epochs = 10
    batches_per_epoch = n_train // batch_size
    
    # Allouer buffers pour batch
    X_batch = libnn.matrix_zeros(batch_size, input_size)
    Y_batch = libnn.matrix_zeros(batch_size, output_size)
    indices = (ctypes.c_int * batch_size)()
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        start_time = time.perf_counter()
        
        for batch in range(batches_per_epoch):
            # Extraire batch
            libnn.mnist_get_batch(train_data, batch_size, X_batch, Y_batch, indices)
            
            # Train
            loss = libnn.network_train_batch(net, X_batch, Y_batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / batches_per_epoch
        epoch_time = time.perf_counter() - start_time
        
        # Accuracy (sur subset pour rapidité)
        subset_size = min(1000, n_train)
        X_test = libnn.matrix_create(subset_size, input_size)
        # Copier subset
        for i in range(subset_size):
            libnn.memcpy(
                ctypes.cast(X_test.contents.data, ctypes.c_void_p).value + i * input_size * 8,
                ctypes.cast(train_data.contents.images.contents.data, ctypes.c_void_p).value + i * input_size * 8,
                input_size * 8
            )
        
        acc = libnn.network_accuracy(net, X_test, train_data.contents.labels, subset_size)
        
        print(f"Epoch {epoch+1}/{n_epochs}  Loss: {avg_loss:.4f}  Acc: {acc*100:.2f}%  Time: {epoch_time:.2f}s")
        
        libnn.matrix_free(X_test)
    
    print("\n✓ Entraînement terminé !")
    
    # Cleanup
    libnn.matrix_free(X_batch)
    libnn.matrix_free(Y_batch)
    libnn.network_free(net)
    libnn.mnist_free(train_data)

if __name__ == "__main__":
    train_mnist()
