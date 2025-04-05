/**
 * @file kernels.cu
 * @brief Implementation of CUDA kernels for SPH simulation.
 *
 * This file implements CUDA kernels for updating the grid, computing density, pressure,
 * forces, integration, and other physics-related computations. Grid helper functions
 * from grid.h are used to reduce code duplication.
 */

#include "kernels.h"
#include "particle.h"
#include "grid.h"

/**
 * @brief CUDA kernel to update grid counters and assign particles to grid cells.
 */
__global__ void updateGrid(const Particle *particles, int N, float cellSize, int gridWidth,
                           int gridHeight, int *gridCounters, unsigned int *gridCells,
                           int maxParticlesPerCell) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Use helper function to compute grid cell index.
        int cellIndex = getCellIndex(particles[i].position, cellSize, gridWidth, gridHeight);
        int index = atomicAdd(&gridCounters[cellIndex], 1);
        if (index < maxParticlesPerCell) {
            gridCells[cellIndex * maxParticlesPerCell + index] = i;
        }
    }
}

/**
 * @brief CUDA kernel to compute density for each particle using grid-based neighbor search.
 */
__global__ void computeDensityGrid(Particle *particles, int N, float cellSize, int gridWidth,
                                   int gridHeight, const int *gridCounters, const unsigned int *gridCells,
                                   int maxParticlesPerCell) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Compute grid cell coordinates for current particle.
        int2 cellCoords = getCellCoordinates(particles[i].position, cellSize, gridWidth, gridHeight);
        float density = 0.0f;
        // Retrieve neighbor cells using helper function.
        int neighbors[9];
        int neighborCount = getNeighborCellIndices(cellCoords.x, cellCoords.y, gridWidth, gridHeight, neighbors);
        for (int n = 0; n < neighborCount; n++) {
            int cellIndex = neighbors[n];
            int count = gridCounters[cellIndex];
            for (int j = 0; j < count; j++) {
                unsigned int neighborIndex = gridCells[cellIndex * maxParticlesPerCell + j];
                float2 r = subtractF2(particles[i].position, particles[neighborIndex].position);
                float rLen = lengthF2(r);
                if (rLen <= cellSize) {
                    density += particles[neighborIndex].mass *
                            (315.0f / (64.0f * PI * powf(cellSize, 9))) * powf((cellSize * cellSize - rLen * rLen), 3);
                }
            }
        }
        particles[i].density = density;
    }
}

/**
 * @brief CUDA kernel to compute pressure for each particle.
 */
__global__ void computePressure(Particle *particles, int N, float K, float RHO0) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].pressure = K * (particles[i].density - RHO0);
    }
}

/**
 * @brief CUDA kernel to compute pressure forces using grid-based neighbor search.
 */
__global__ void computePressureForcesGrid(Particle *particles, int N, float cellSize, int gridWidth,
                                          int gridHeight, const int *gridCounters, const unsigned int *gridCells,
                                          int maxParticlesPerCell) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 force = {0.0f, 0.0f};
        int2 cellCoords = getCellCoordinates(particles[i].position, cellSize, gridWidth, gridHeight);
        int neighbors[9];
        int neighborCount = getNeighborCellIndices(cellCoords.x, cellCoords.y, gridWidth, gridHeight, neighbors);
        for (int n = 0; n < neighborCount; n++) {
            int cellIndex = neighbors[n];
            int count = gridCounters[cellIndex];
            for (int j = 0; j < count; j++) {
                unsigned int neighborIndex = gridCells[cellIndex * maxParticlesPerCell + j];
                if (neighborIndex == i) continue;
                float2 r = subtractF2(particles[i].position, particles[neighborIndex].position);
                float rLen = lengthF2(r);
                if (rLen <= cellSize && rLen > 0.0f) {
                    float gradCoefficient = (-45.0f / (PI * powf(cellSize, 6))) * powf((cellSize - rLen), 2) / rLen;
                    float2 grad = {r.x * gradCoefficient, r.y * gradCoefficient};
                    float term = (particles[i].pressure + particles[neighborIndex].pressure) /
                                 (2.0f * particles[neighborIndex].density);
                    force.x += -particles[neighborIndex].mass * term * grad.x;
                    force.y += -particles[neighborIndex].mass * term * grad.y;
                }
            }
        }
        particles[i].force.x += force.x;
        particles[i].force.y += force.y;
    }
}

/**
 * @brief CUDA kernel to compute viscosity forces using grid-based neighbor search.
 */
__global__ void computeViscosityForcesGrid(Particle *particles, int N, float cellSize, int gridWidth,
                                           int gridHeight, const int *gridCounters, const unsigned int *gridCells,
                                           int maxParticlesPerCell, float MU) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 force = {0.0f, 0.0f};
        int2 cellCoords = getCellCoordinates(particles[i].position, cellSize, gridWidth, gridHeight);
        int neighbors[9];
        int neighborCount = getNeighborCellIndices(cellCoords.x, cellCoords.y, gridWidth, gridHeight, neighbors);
        for (int n = 0; n < neighborCount; n++) {
            int cellIndex = neighbors[n];
            int count = gridCounters[cellIndex];
            for (int j = 0; j < count; j++) {
                unsigned int neighborIndex = gridCells[cellIndex * maxParticlesPerCell + j];
                if (neighborIndex == i) continue;
                float2 r = subtractF2(particles[i].position, particles[neighborIndex].position);
                float rLen = lengthF2(r);
                if (rLen <= cellSize) {
                    float laplacian = (45.0f / (PI * powf(cellSize, 6))) * (cellSize - rLen);
                    float2 diff = {
                        particles[neighborIndex].velocity.x - particles[i].velocity.x,
                        particles[neighborIndex].velocity.y - particles[i].velocity.y
                    };
                    force.x += MU * particles[neighborIndex].mass * diff.x / particles[neighborIndex].density *
                            laplacian;
                    force.y += MU * particles[neighborIndex].mass * diff.y / particles[neighborIndex].density *
                            laplacian;
                }
            }
        }
        particles[i].force.x += force.x;
        particles[i].force.y += force.y;
    }
}

/**
 * @brief CUDA kernel to apply gravity to each particle.
 */
__global__ void applyGravity(Particle *particles, int N, float GRAVITY) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].force.y += GRAVITY * particles[i].density;
    }
}

/**
 * @brief CUDA kernel to save the current position of each particle.
 */
__global__ void saveOldPosition(Particle *particles, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].oldPosition = particles[i].position;
    }
}

/**
 * @brief CUDA kernel to predict new positions of particles based on current velocities.
 */
__global__ void predictPosition(Particle *particles, int N, float lookAhead) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].position.x += particles[i].velocity.x * lookAhead;
        particles[i].position.y += particles[i].velocity.y * lookAhead;
    }
}

/**
 * @brief CUDA kernel to integrate particle motion using Euler integration.
 */
__global__ void integrate(Particle *particles, int N, float dt) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float ax = particles[i].force.x / particles[i].density;
        float ay = particles[i].force.y / particles[i].density;
        particles[i].velocity.x += ax * dt;
        particles[i].velocity.y += ay * dt;
        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        particles[i].force.x = 0.0f;
        particles[i].force.y = 0.0f;
    }
}

/**
 * @brief CUDA kernel to reset particle positions to their saved old positions.
 */
__global__ void resetToOldPosition(Particle *particles, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].position = particles[i].oldPosition;
    }
}

/**
 * @brief CUDA kernel to apply boundary conditions by reflecting particles at domain boundaries.
 */
__global__ void applyBoundaryConditions(Particle *particles, int N, float dimX, float dimY,
                                        float damping, float radius) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (particles[i].position.x < radius) {
            particles[i].velocity.x *= damping;
            particles[i].position.x = radius;
        }
        if (particles[i].position.x > dimX - radius) {
            particles[i].velocity.x *= damping;
            particles[i].position.x = dimX - radius;
        }
        if (particles[i].position.y < radius) {
            particles[i].velocity.y *= damping;
            particles[i].position.y = radius;
        }
        if (particles[i].position.y > dimY - radius) {
            particles[i].velocity.y *= damping;
            particles[i].position.y = dimY - radius;
        }
    }
}

/**
 * @brief CUDA kernel to apply a push force from the mouse to particles.
 */
__global__ void applyPushForce(Particle *particles, int N, float2 mousePos, float strength,
                               float interactionRadius) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 direction = {mousePos.x - particles[i].position.x, mousePos.y - particles[i].position.y};
        float distance = lengthF2(direction);
        if (distance < interactionRadius) {
            if (distance > interactionRadius * 0.5f || strength < 0.0f) {
                // Normalize direction
                direction.x /= distance;
                direction.y /= distance;
                float forceMag = strength / (distance + 1.0f);
                particles[i].force.x += forceMag * direction.x;
                particles[i].force.y += forceMag * direction.y;
            }
        }
    }
}

/**
 * @brief CUDA kernel to apply a swirling force from the mouse to particles.
 */
__global__ void applySwirlForce(Particle *particles, int N, float2 mousePos, float strength,
                                float interactionRadius) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 diff = {particles[i].position.x - mousePos.x, particles[i].position.y - mousePos.y};
        float distance = lengthF2(diff);
        if (distance < interactionRadius && distance > 0.0f) {
            // Normalize diff vector
            float2 normDiff = {diff.x / distance, diff.y / distance};
            // Compute perpendicular vector (rotated 90 degrees clockwise)
            float2 perp = {-normDiff.y, normDiff.x};
            float forceMag = strength / (distance + 1.0f);
            particles[i].force.x += forceMag * perp.x;
            particles[i].force.y += forceMag * perp.y;
        }
    }
}
