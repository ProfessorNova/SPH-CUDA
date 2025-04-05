/**
* @file particle.h
 * @brief Definition of the Particle structure and helper functions.
 *
 * This file defines the Particle data structure and inline functions for operations on float2 types.
 */
#ifndef PARTICLE_H
#define PARTICLE_H

#include <raylib.h>
#include <cuda_runtime.h>

/**
 * @brief Structure representing a particle in the simulation.
 */
struct Particle {
    float2 position; // Position in simulation space
    float2 oldPosition; // Previous position (for stable integration)
    float2 velocity; // Velocity vector
    float2 force; // Accumulated force
    float mass; // Mass of the particle
    float density; // Computed density
    float pressure; // Computed pressure
};

/**
 * @brief Compute the Euclidean length of a float2 vector.
 *
 * @param v A float2 vector.
 * @return float The length of the vector.
 */
__device__ __host__ inline float lengthF2(const float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

/**
 * @brief Subtract two float2 vectors.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return float2 The result of a - b.
 */
__device__ __host__ inline float2 subtractF2(const float2 a, const float2 b) {
    return {a.x - b.x, a.y - b.y};
}

#endif // PARTICLE_H
