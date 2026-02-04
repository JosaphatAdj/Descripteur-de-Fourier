#!/usr/bin/env python3
"""
Télécharge le dataset MNIST
===========================

Source: http://yann.lecun.com/exdb/mnist/
"""

import urllib.request
import gzip
import os
from pathlib import Path

MNIST_URL = "http://yann.lecun.com/exdb/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

def download_mnist(data_dir="data/mnist"):
    """Télécharge et décompresse MNIST."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for filename in FILES:
        gz_path = data_path / filename
        final_path = data_path / filename[:-3]  # Sans .gz
        
        if final_path.exists():
            print(f"✓ {filename[:-3]} déjà présent")
            continue
        
        print(f"Téléchargement {filename}...")
        url = MNIST_URL + filename
        urllib.request.urlretrieve(url, str(gz_path))
        
        print(f"  Décompression...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(final_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        gz_path.unlink()  # Supprimer .gz
        print(f"✓ {filename[:-3]} prêt")
    
    print("\n✓ Dataset MNIST téléchargé et prêt !")

if __name__ == "__main__":
    download_mnist()
