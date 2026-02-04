/**
 * Neural Network - Forward & Backward Pass
 * =========================================
 * 
 * Utilise MASSIVEMENT cblas_dgemm (6 appels par mini-batch)
 */

#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// ============================================
// Initialisation
// ============================================

Network* network_create(int input_size, int hidden_size, int output_size,
                       double learning_rate, int batch_size) {
    Network* net = (Network*)malloc(sizeof(Network));
    if (!net) return NULL;
    
    // Initialisation Xavier/He pour poids
    double std_w1 = sqrt(2.0 / input_size);
    double std_w2 = sqrt(2.0 / hidden_size);
    
    net->W1 = matrix_random_normal(input_size, hidden_size, 0.0, std_w1);
    net->b1 = matrix_zeros(1, hidden_size);
    net->W2 = matrix_random_normal(hidden_size, output_size, 0.0, std_w2);
    net->b2 = matrix_zeros(1, output_size);
    
    // Cache pour backward
    net->Z1 = matrix_zeros(batch_size, hidden_size);
    net->A1 = matrix_zeros(batch_size, hidden_size);
    net->Z2 = matrix_zeros(batch_size, output_size);
    net->A2 = matrix_zeros(batch_size, output_size);
    
    // Gradients
    net->dW1 = matrix_zeros(input_size, hidden_size);
    net->db1 = matrix_zeros(1, hidden_size);
    net->dW2 = matrix_zeros(hidden_size, output_size);
    net->db2 = matrix_zeros(1, output_size);
    
    net->learning_rate = learning_rate;
    net->batch_size = batch_size;
    
    return net;
}

void network_free(Network* net) {
    if (!net) return;
    
    matrix_free(net->W1);
    matrix_free(net->b1);
    matrix_free(net->W2);
    matrix_free(net->b2);
    matrix_free(net->Z1);
    matrix_free(net->A1);
    matrix_free(net->Z2);
    matrix_free(net->A2);
    matrix_free(net->dW1);
    matrix_free(net->db1);
    matrix_free(net->dW2);
    matrix_free(net->db2);
    
    free(net);
}

// ============================================
// Forward Pass (2 GEMM)
// ============================================

void network_forward(Network* net, const Matrix* X) {
    /**
     * Forward propagation
     * 
     * Layer 1:
     *   Z1 = X·W1           <- GEMM #1 (batch × 784) · (784 × 128)
     *   Z1 += b1            (broadcasting)
     *   A1 = ReLU(Z1)
     * 
     * Layer 2:
     *   Z2 = A1·W2          <- GEMM #2 (batch × 128) · (128 × 10)
     *   Z2 += b2
     *   A2 = Softmax(Z2)
     */
    
    // Layer 1: Z1 = X·W1
    matrix_multiply(X, net->W1, net->Z1, 1.0, 0.0, 0, 0);  // GEMM #1
    
    // Z1 += b1 (broadcast)
    matrix_add_vector_broadcast(net->Z1, net->b1, 1);
    
    // A1 = ReLU(Z1)
    matrix_copy(net->Z1, net->A1);
    matrix_relu(net->A1);
    
    // Layer 2: Z2 = A1·W2
    matrix_multiply(net->A1, net->W2, net->Z2, 1.0, 0.0, 0, 0);  // GEMM #2
    
    // Z2 += b2
    matrix_add_vector_broadcast(net->Z2, net->b2, 1);
    
    // A2 = Softmax(Z2)
    matrix_copy(net->Z2, net->A2);
    matrix_softmax(net->A2);
}

// ============================================
// Backward Pass (3 GEMM)
// ============================================

void network_backward(Network* net, const Matrix* X, const Matrix* dY) {
    /**
     * Backward propagation
     * 
     * dY est le gradient de la loss: dY = A2 - Y_true (softmax + cross-entropy)
     * 
     * Layer 2:
     *   dW2 = A1ᵀ·dY       <- GEMM #3 (128 × batch) · (batch × 10)
     *   db2 = sum(dY, axis=0)
     *   dA1 = dY·W2ᵀ       <- GEMM #4 (batch × 10) · (10 × 128)
     * 
     * ReLU gradient:
     *   dZ1 = dA1 .* (Z1 > 0)
     * 
     * Layer 1:
     *   dW1 = Xᵀ·dZ1       <- GEMM #5 (784 × batch) · (batch × 128)
     *   db1 = sum(dZ1, axis=0)
     */
    
    int batch_size = dY->rows;
    
    // ===== Backward Layer 2 =====
    
    // dW2 = A1ᵀ·dY (transpose A1)
    matrix_multiply(net->A1, dY, net->dW2, 1.0, 0.0, 1, 0);  // GEMM #3
    
    // db2 = sum(dY, axis=0) = moyenne sur batch
    matrix_fill(net->db2, 0.0);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dY->cols; j++) {
            net->db2->data[j] += dY->data[i * dY->cols + j];
        }
    }
    matrix_scale(1.0 / batch_size, net->db2);
    
    // dA1 = dY·W2ᵀ (transpose W2)
    Matrix* dA1 = matrix_zeros(batch_size, net->W2->rows);
    matrix_multiply(dY, net->W2, dA1, 1.0, 0.0, 0, 1);  // GEMM #4
    
    // ===== ReLU Gradient =====
    
    // dZ1 = dA1 .* (Z1 > 0)
    Matrix* dZ1 = matrix_zeros(batch_size, net->Z1->cols);
    matrix_copy(dA1, dZ1);
    matrix_relu_derivative(net->Z1, dZ1);
    
    // ===== Backward Layer 1 =====
    
    // dW1 = Xᵀ·dZ1 (transpose X)
    matrix_multiply(X, dZ1, net->dW1, 1.0, 0.0, 1, 0);  // GEMM #5
    
    // db1 = sum(dZ1, axis=0)
    matrix_fill(net->db1, 0.0);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dZ1->cols; j++) {
            net->db1->data[j] += dZ1->data[i * dZ1->cols + j];
        }
    }
    matrix_scale(1.0 / batch_size, net->db1);
    
    // Cleanup temporaires
    matrix_free(dA1);
    matrix_free(dZ1);
}

// ============================================
// Gradient Descent (BLAS Level 1)
// ============================================

void network_update_weights(Network* net) {
    /**
     * W -= lr * dW
     * Utilise cblas_daxpy (BLAS Level 1)
     */
    double neg_lr = -net->learning_rate;
    
    matrix_axpy(neg_lr, net->dW1, net->W1);
    matrix_axpy(neg_lr, net->db1, net->b1);
    matrix_axpy(neg_lr, net->dW2, net->W2);
    matrix_axpy(neg_lr, net->db2, net->b2);
}

// ============================================
// Loss Functions
// ============================================

double cross_entropy_loss(const Matrix* Y_pred, const Matrix* Y_true, Matrix* dY) {
    /**
     * Cross-Entropy Loss + Gradient
     * 
     * Loss: -sum(Y_true * log(Y_pred)) / batch_size
     * Gradient: dY = Y_pred - Y_true (softmax + cross-entropy simplifié!)
     */
    int batch_size = Y_pred->rows;
    int n_classes = Y_pred->cols;
    
    double loss = 0.0;
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_classes; j++) {
            int idx = i * n_classes + j;
            
            // Loss
            if (Y_true->data[idx] > 0.5) {  // one-hot = 1
                // Clamp pour stabilité: log(max(1e-10, pred))
                double pred = Y_pred->data[idx];
                if (pred < 1e-10) pred = 1e-10;
                loss -= log(pred);
            }
            
            // Gradient: dY = Y_pred - Y_true
            dY->data[idx] = Y_pred->data[idx] - Y_true->data[idx];
        }
    }
    
    return loss / batch_size;
}

// ============================================
// Training
// ============================================

double network_train_batch(Network* net, const Matrix* X, const Matrix* Y_true) {
    /**
     * Train sur un mini-batch
     * 
     * 1. Forward
     * 2. Compute loss + gradient
     * 3. Backward
     * 4. Update weights
     */
    
    // Forward
    network_forward(net, X);
    
    // Loss + gradient
    Matrix* dY = matrix_zeros(Y_true->rows, Y_true->cols);
    double loss = cross_entropy_loss(net->A2, Y_true, dY);
    
    // Backward
    network_backward(net, X, dY);
    
    // Update
    network_update_weights(net);
    
    matrix_free(dY);
    
    return loss;
}

// ============================================
// Prédiction
// ============================================

void network_predict(Network* net, const Matrix* X, int* predictions) {
    /**
     * Prédit les labels (argmax de A2)
     */
    network_forward(net, X);
    
    for (int i = 0; i < X->rows; i++) {
        double max_val = net->A2->data[i * net->A2->cols];
        int max_idx = 0;
        
        for (int j = 1; j < net->A2->cols; j++) {
            double val = net->A2->data[i * net->A2->cols + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        predictions[i] = max_idx;
    }
}

double network_accuracy(Network* net, const Matrix* X, const int* Y_true, int n_samples) {
    /**
     * Calcule l'accuracy
     */
    int* predictions = (int*)malloc(n_samples * sizeof(int));
    network_predict(net, X, predictions);
    
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        if (predictions[i] == Y_true[i]) {
            correct++;
        }
    }
    
    free(predictions);
    
    return (double)correct / n_samples;
}
