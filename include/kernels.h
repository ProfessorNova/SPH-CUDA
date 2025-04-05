/**
 * @file kernels.h
 * @brief Declarations of CUDA kernels for simulation.
 *
 * This file declares the CUDA kernels for updating the grid, computing density, pressure,
 * forces, integration, and other physics-related computations.
 */
#ifndef KERNELS_H
#define KERNELS_H

#include "particle.h"

/**
 * @brief CUDA kernel to update grid counters and assign particles to grid cells.
 */
__global__ void updateGrid(const Particle *particles, int N, float cellSize, int gridWidth,
                           int gridHeight, int *gridCounters, unsigned int *gridCells,
                           int maxParticlesPerCell);

/**
 * @brief CUDA kernel to compute density for each particle using grid hashing.
 */
__global__ void computeDensityGrid(Particle *particles, int N, float cellSize, int gridWidth,
                                   int gridHeight, const int *gridCounters, const unsigned int *gridCells,
                                   int maxParticlesPerCell);

/**
 * @brief CUDA kernel to compute pressure for each particle.
 */
__global__ void computePressure(Particle *particles, int N, float K, float RHO0);

/**
 * @brief CUDA kernel to compute pressure forces using grid-based neighbor search.
 */
__global__ void computePressureForcesGrid(Particle *particles, int N, float cellSize, int gridWidth,
                                          int gridHeight, const int *gridCounters, const unsigned int *gridCells,
                                          int maxParticlesPerCell);

/**
 * @brief CUDA kernel to compute viscosity forces using grid-based neighbor search.
 */
__global__ void computeViscosityForcesGrid(Particle *particles, int N, float cellSize, int gridWidth,
                                           int gridHeight, const int *gridCounters, const unsigned int *gridCells,
                                           int maxParticlesPerCell, float MU);

/**
 * @brief CUDA kernel to apply gravity to each particle.
 */
__global__ void applyGravity(Particle *particles, int N, float GRAVITY);

/**
 * @brief CUDA kernel to save the current position of each particle.
 */
__global__ void saveOldPosition(Particle *particles, int N);

/**
 * @brief CUDA kernel to predict the new position of each particle.
 */
__global__ void predictPosition(Particle *particles, int N, float lookAhead);

/**
 * @brief CUDA kernel to integrate particle motion using Euler integration.
 */
__global__ void integrate(Particle *particles, int N, float dt);

/**
 * @brief CUDA kernel to reset particle positions to their saved old positions.
 */
__global__ void resetToOldPosition(Particle *particles, int N);

/**
 * @brief CUDA kernel to apply boundary conditions by reflecting particles at domain boundaries.
 */
__global__ void applyBoundaryConditions(Particle *particles, int N, float dimX, float dimY,
                                        float damping, float radius);

/**
 * @brief CUDA kernel to apply a push force from the mouse to particles.
 */
__global__ void applyPushForce(Particle *particles, int N, float2 mousePos, float strength,
                               float interactionRadius);

/**
 * @brief CUDA kernel to apply a swirling force from the mouse to particles.
 */
__global__ void applySwirlForce(Particle *particles, int N, float2 mousePos, float strength,
                                float interactionRadius);

#endif // KERNELS_H
