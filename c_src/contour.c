/**
 * contour.c - Utilitaires pour la gestion des contours
 * =====================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/fourier.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/**
 * Crée un contour à partir de tableaux de coordonnées.
 */
Contour* contour_create(const double* x, const double* y, size_t n) {
    if (!x || !y || n == 0) {
        return NULL;
    }
    
    Contour* contour = (Contour*)malloc(sizeof(Contour));
    if (!contour) {
        return NULL;
    }
    
    contour->points = (Point2D*)malloc(n * sizeof(Point2D));
    if (!contour->points) {
        free(contour);
        return NULL;
    }
    
    for (size_t i = 0; i < n; i++) {
        contour->points[i].x = x[i];
        contour->points[i].y = y[i];
    }
    
    contour->n_points = n;
    return contour;
}


/**
 * Libère la mémoire d'un contour.
 */
void contour_free(Contour* contour) {
    if (contour) {
        if (contour->points) {
            free(contour->points);
        }
        free(contour);
    }
}


/**
 * Génère un cercle de test.
 */
Contour* contour_create_circle(int n_points, double radius) {
    if (n_points <= 0 || radius <= 0) {
        return NULL;
    }
    
    Contour* contour = (Contour*)malloc(sizeof(Contour));
    if (!contour) {
        return NULL;
    }
    
    contour->points = (Point2D*)malloc(n_points * sizeof(Point2D));
    if (!contour->points) {
        free(contour);
        return NULL;
    }
    
    for (int i = 0; i < n_points; i++) {
        double theta = 2.0 * M_PI * i / n_points;
        contour->points[i].x = radius * cos(theta);
        contour->points[i].y = radius * sin(theta);
    }
    
    contour->n_points = n_points;
    return contour;
}


/**
 * Génère un carré de test.
 */
Contour* contour_create_square(int n_points_per_side, double side_length) {
    if (n_points_per_side <= 0 || side_length <= 0) {
        return NULL;
    }
    
    int total_points = 4 * n_points_per_side;
    double half_side = side_length / 2.0;
    
    Contour* contour = (Contour*)malloc(sizeof(Contour));
    if (!contour) {
        return NULL;
    }
    
    contour->points = (Point2D*)malloc(total_points * sizeof(Point2D));
    if (!contour->points) {
        free(contour);
        return NULL;
    }
    
    int idx = 0;
    double step = side_length / n_points_per_side;
    
    /* Côté bas (gauche à droite) */
    for (int i = 0; i < n_points_per_side; i++) {
        contour->points[idx].x = -half_side + i * step;
        contour->points[idx].y = -half_side;
        idx++;
    }
    
    /* Côté droit (bas en haut) */
    for (int i = 0; i < n_points_per_side; i++) {
        contour->points[idx].x = half_side;
        contour->points[idx].y = -half_side + i * step;
        idx++;
    }
    
    /* Côté haut (droite à gauche) */
    for (int i = 0; i < n_points_per_side; i++) {
        contour->points[idx].x = half_side - i * step;
        contour->points[idx].y = half_side;
        idx++;
    }
    
    /* Côté gauche (haut en bas) */
    for (int i = 0; i < n_points_per_side; i++) {
        contour->points[idx].x = -half_side;
        contour->points[idx].y = half_side - i * step;
        idx++;
    }
    
    contour->n_points = total_points;
    return contour;
}


/**
 * Alloue un tableau de descripteurs.
 */
FourierDescriptors* descriptors_create(size_t n) {
    FourierDescriptors* desc = (FourierDescriptors*)malloc(sizeof(FourierDescriptors));
    if (!desc) {
        return NULL;
    }
    
    desc->values = (double*)calloc(n, sizeof(double));
    if (!desc->values) {
        free(desc);
        return NULL;
    }
    
    desc->n_values = n;
    return desc;
}


/**
 * Libère un tableau de descripteurs.
 */
void descriptors_free(FourierDescriptors* desc) {
    if (desc) {
        if (desc->values) {
            free(desc->values);
        }
        free(desc);
    }
}
