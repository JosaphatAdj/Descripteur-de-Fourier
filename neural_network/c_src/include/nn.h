#ifndef NN_H
#define NN_H

#include "matrix.h"

/**
 * Neural Network Dense (Fully Connected)
 * =======================================
 * 
 * Architecture: Input(784) -> Hidden(128, ReLU) -> Output(10, Softmax)
 * Optimisé avec OpenBLAS GEMM pour forward et backward pass.
 */

// Structure du réseau
typedef struct {
    // Couche 1: 784 -> 128
    Matrix* W1;     // Poids (784 × 128)
    Matrix* b1;     // Biais (1 × 128)
    
    // Couche 2: 128 -> 10
    Matrix* W2;     // Poids (128 × 10)
    Matrix* b2;     // Biais (1 × 10)
    
    // Cache pour backward pass
    Matrix* Z1;     // Pré-activation couche 1 (batch_size × 128)
    Matrix* A1;     // Post-activation (ReLU) couche 1
    Matrix* Z2;     // Pré-activation couche 2 (batch_size × 10)
    Matrix* A2;     // Post-activation (Softmax) couche 2
    
    // Gradients
    Matrix* dW1, *db1;
    Matrix* dW2, *db2;
    
    // Hyperparamètres
    double learning_rate;
    int batch_size;
} Network;

// ====================
// Initialisation/Destruction
// ====================

Network* network_create(int input_size, int hidden_size, int output_size,
                       double learning_rate, int batch_size);
void network_free(Network* net);

// ====================
// Forward Pass (utilise GEMM massivement)
// ====================

/**
 * Forward propagation sur un mini-batch
 * Input: X (batch_size × 784)
 * Output: predictions (batch_size × 10) dans net->A2
 */
void network_forward(Network* net, const Matrix* X);

// ====================
// Backward Pass (utilise GEMM massivement)
// ====================

/**
 * Backward propagation + calcul des gradients
 * dY: gradient de la loss (batch_size × 10)
 * X: input original (batch_size × 784)
 */
void network_backward(Network* net, const Matrix* X, const Matrix* dY);

// ====================
// Mise à jour des poids (Gradient Descent)
// ====================

/**
 * Applique gradient descent: W -= lr * dW
 * Utilise cblas_daxpy (BLAS Level 1)
 */
void network_update_weights(Network* net);

// ====================
// Training
// ====================

/**
 * Entraîne le réseau sur un mini-batch
 * Retourne la loss moyenne
 */
double network_train_batch(Network* net, const Matrix* X, const Matrix* Y_true);

// ====================
// Évaluation
// ====================

/**
 * Prédit les labels pour un batch
 * Retourne un vecteur de labels (argmax de A2)
 */
void network_predict(Network* net, const Matrix* X, int* predictions);

/**
 * Calcule l'accuracy sur un dataset
 */
double network_accuracy(Network* net, const Matrix* X, const int* Y_true, int n_samples);

// ====================
// Loss Functions
// ====================

/**
 * Cross-Entropy Loss + gradient
 * dY = A2 - Y_true (one-hot)
 */
double cross_entropy_loss(const Matrix* Y_pred, const Matrix* Y_true, Matrix* dY);

#endif // NN_H
