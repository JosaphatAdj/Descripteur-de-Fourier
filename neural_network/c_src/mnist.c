/**
 * Chargement du dataset MNIST
 * ============================
 * 
 * Format IDX (binary) du site de Yann LeCun
 */

#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Utilitaire: lire entier big-endian
static uint32_t read_int32(FILE* file) {
    uint8_t bytes[4];
    fread(bytes, 1, 4, file);
    return ((uint32_t)bytes[0] << 24) |
           ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) |
           ((uint32_t)bytes[3]);
}

MNISTDataset* mnist_load(const char* images_path, const char* labels_path) {
    FILE* img_file = fopen(images_path, "rb");
    FILE* lbl_file = fopen(labels_path, "rb");
    
    if (!img_file || !lbl_file) {
        fprintf(stderr, "Erreur: Impossible d'ouvrir les fichiers MNIST\n");
        if (img_file) fclose(img_file);
        if (lbl_file) fclose(lbl_file);
        return NULL;
    }
    
    // Lire headers images
    uint32_t magic = read_int32(img_file);
    uint32_t n_images = read_int32(img_file);
    uint32_t rows = read_int32(img_file);
    uint32_t cols = read_int32(img_file);
    
    if (magic != 2051 || rows != 28 || cols != 28) {
        fprintf(stderr, "Erreur: Format MNIST images invalide\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }
    
    // Lire header labels
    magic = read_int32(lbl_file);
    uint32_t n_labels = read_int32(lbl_file);
    
    if (magic != 2049 || n_labels != n_images) {
        fprintf(stderr, "Erreur: Format MNIST labels invalide\n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }
    
    // Allouer dataset
    MNISTDataset* dataset = (MNISTDataset*)malloc(sizeof(MNISTDataset));
    dataset->n_samples = n_images;
    dataset->images = matrix_create(n_images, 784);  // 28×28 = 784
    dataset->labels = (int*)malloc(n_images * sizeof(int));
    
    // Lire images
    uint8_t* buffer = (uint8_t*)malloc(784);
    for (uint32_t i = 0; i < n_images; i++) {
        fread(buffer, 1, 784, img_file);
        
        // Normaliser [0, 255] -> [0, 1]
        for (int j = 0; j < 784; j++) {
            dataset->images->data[i * 784 + j] = buffer[j] / 255.0;
        }
    }
    free(buffer);
    
    // Lire labels
    uint8_t label;
    for (uint32_t i = 0; i < n_labels; i++) {
        fread(&label, 1, 1, lbl_file);
        dataset->labels[i] = (int)label;
    }
    
    fclose(img_file);
    fclose(lbl_file);
    
    printf("✓ MNIST chargé: %u images\n", n_images);
    
    return dataset;
}

MNISTDataset* mnist_load_train(const char* data_dir) {
    char img_path[512], lbl_path[512];
    snprintf(img_path, 512, "%s/train-images-idx3-ubyte", data_dir);
    snprintf(lbl_path, 512, "%s/train-labels-idx1-ubyte", data_dir);
    return mnist_load(img_path, lbl_path);
}

MNISTDataset* mnist_load_test(const char* data_dir) {
    char img_path[512], lbl_path[512];
    snprintf(img_path, 512, "%s/t10k-images-idx3-ubyte", data_dir);
    snprintf(lbl_path, 512, "%s/t10k-labels-idx1-ubyte", data_dir);
    return mnist_load(img_path, lbl_path);
}

void mnist_free(MNISTDataset* dataset) {
    if (dataset) {
        matrix_free(dataset->images);
        free(dataset->labels);
        free(dataset);
    }
}

void mnist_get_batch(const MNISTDataset* dataset, int batch_size,
                     Matrix* X, Matrix* Y, int* indices) {
    /**
     * Extrait un mini-batch aléatoire
     */
    
    // Générer indices aléatoires
    for (int i = 0; i < batch_size; i++) {
        indices[i] = rand() % dataset->n_samples;
    }
    
    // Copier images dans X
    for (int i = 0; i < batch_size; i++) {
        int idx = indices[i];
        memcpy(X->data + i * 784,
               dataset->images->data + idx * 784,
               784 * sizeof(double));
    }
    
    // Créer one-hot Y
    matrix_fill(Y, 0.0);
    for (int i = 0; i < batch_size; i++) {
        int label = dataset->labels[indices[i]];
        Y->data[i * 10 + label] = 1.0;
    }
}

void labels_to_one_hot(const int* labels, int n_samples, Matrix* one_hot) {
    matrix_fill(one_hot, 0.0);
    for (int i = 0; i < n_samples; i++) {
        one_hot->data[i * 10 + labels[i]] = 1.0;
    }
}
