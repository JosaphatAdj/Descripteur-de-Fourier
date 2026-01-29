# Dockerfile pour le projet Descripteurs de Fourier avec OpenBLAS
# Architecture: C (OpenBLAS) + Python (wrappers/visualisation)

FROM python:3.11-slim-bookworm

# ============================================================
# INSTALLATION DES DÉPENDANCES C/OPENBLAS
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compilation C
    build-essential \
    gcc \
    gfortran \
    make \
    cmake \
    # OpenBLAS et LAPACK
    libopenblas-dev \
    liblapack-dev \
    # FFTW pour les FFT (optionnel mais recommandé)
    libfftw3-dev \
    # Outils
    pkg-config \
    git \
    wget \
    valgrind \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# CONFIGURATION OPENBLAS
# ============================================================
ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
# Chemins pour le linker
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ============================================================
# RÉPERTOIRE DE TRAVAIL
# ============================================================
WORKDIR /app

# ============================================================
# DÉPENDANCES PYTHON (wrappers + visualisation)
# ============================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# COPIE DU CODE SOURCE
# ============================================================
COPY . .

# ============================================================
# COMPILATION DU CODE C
# ============================================================
RUN cd /app/c_src && make clean && make all

# Point d'entrée par défaut
CMD ["python", "python/main.py"]
