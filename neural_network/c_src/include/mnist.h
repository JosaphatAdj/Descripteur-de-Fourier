#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"

/**
 * Chargement du dataset MNIST
 * ============================
 * 
 * Format: Images 28×28 pixels, labels 0-9
 * Source: http://yann.lecun.com/exdb/mnist/
 */

// Structure dataset
typedef struct {
    Matrix* images;     // (n_samples × 784) normalisées [0, 1]
    int* labels;        // (n_samples,) valeurs 0-9
    int n_samples;
} MNISTDataset;

/**
 * Charge MNIST depuis fichiers binaires
 * 
 * Fichiers attendus:
 * - train-images-idx3-ubyte
 * - train-labels-idx1-ubyte
 * - t10k-images-idx3-ubyte
 * - t10k-labels-idx1-ubyte
 */
MNISTDataset* mnist_load_train(const char* data_dir);
MNISTDataset* mnist_load_test(const char* data_dir);

/**
 * Libère le dataset
 */
void mnist_free(MNISTDataset* dataset);

/**
 * Extrait un mini-batch aléatoire
 * Remplit X (batch_size × 784) et Y (batch_size × 10 one-hot)
 */
void mnist_get_batch(const MNISTDataset* dataset, int batch_size,
                     Matrix* X, Matrix* Y, int* indices);

/**
 * Convertit labels en one-hot encoding
 * Input: labels[batch_size] (0-9)
 * Output: one_hot (batch_size × 10)
 */
void labels_to_one_hot(const int* labels, int n_samples, Matrix* one_hot);

#endif // MNIST_H
