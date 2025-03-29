/**
 * @file main.cu
 * @brief Example application using raylib for rendering and CUDA kernels to update particles and calculate colors.
 */

#include "raylib.h"
#include <iostream>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Helper functions for both host and device

/**
 * @brief Creates a Color from RGBA components.
 *
 * @param r Red component.
 * @param g Green component.
 * @param b Blue component.
 * @param a Alpha component.
 * @return Color The created color.
 */
__device__ __host__ Color makeColor(const unsigned char r, const unsigned char g, const unsigned char b,
                                    const unsigned char a) {
    Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    c.a = a;
    return c;
}

/**
 * @brief Linearly interpolates between two float values.
 *
 * @param a Starting value.
 * @param b Ending value.
 * @param t Interpolation factor [0,1].
 * @return float The interpolated value.
 */
__device__ __host__ float lerpFloat(const float a, const float b, const float t) {
    return a + (b - a) * t;
}

/**
 * @brief Linearly interpolates between two Colors.
 *
 * @param c1 The starting color.
 * @param c2 The ending color.
 * @param t Interpolation factor [0,1].
 * @return Color The interpolated color.
 */
__device__ __host__ Color lerpColor(const Color c1, const Color c2, const float t) {
    Color result;
    result.r = static_cast<unsigned char>(lerpFloat(c1.r, c2.r, t));
    result.g = static_cast<unsigned char>(lerpFloat(c1.g, c2.g, t));
    result.b = static_cast<unsigned char>(lerpFloat(c1.b, c2.b, t));
    result.a = 255;
    return result;
}

/**
 * @brief Computes a color based on the particle speed.
 *
 * Uses piecewise interpolation among 3 segments:
 *   BLUE -> GREEN -> YELLOW -> RED.
 *
 * @param speed The particle's speed.
 * @param maxVel Maximum velocity for normalization.
 * @return Color The computed color.
 */
__device__ __host__ Color getVelocityColor(const float speed, const float maxVel) {
    float t = speed / maxVel;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    // Piecewise interpolation:
    // BLUE (0.0) -> GREEN (0.33) -> YELLOW (0.66) -> RED (1.0)
    if (t < 0.33f) {
        const float localT = t / 0.33f;
        return lerpColor(makeColor(0, 0, 255, 255), makeColor(0, 255, 0, 255), localT);
    } else if (t < 0.66f) {
        const float localT = (t - 0.33f) / 0.33f;
        return lerpColor(makeColor(0, 255, 0, 255), makeColor(255, 255, 0, 255), localT);
    } else {
        const float localT = (t - 0.66f) / 0.34f;
        return lerpColor(makeColor(255, 255, 0, 255), makeColor(255, 0, 0, 255), localT);
    }
}

//---------------------------------------------------------------------
// Particle structure

/**
 * @brief Structure representing a particle.
 *
 * Contains position, velocity (both as float2), and mass.
 */
struct Particle {
    float2 position;
    float2 velocity;
    float mass;
};

//---------------------------------------------------------------------
// CUDA kernels

/**
 * @brief CUDA kernel to update particle positions.
 *
 * Each thread updates one particle by adding its velocity to its position.
 *
 * @param particles Pointer to an array of Particle structures in unified memory.
 * @param numParticles Total number of particles.
 */
__global__ void updateParticles(Particle *particles, const int numParticles) {
    if (const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numParticles) {
        particles[idx].position.x += particles[idx].velocity.x;
        particles[idx].position.y += particles[idx].velocity.y;
    }
}

/**
 * @brief CUDA kernel to calculate the color for each particle based on its velocity.
 *
 * Each thread computes the speed of one particle and maps it to a color.
 *
 * @param particles Pointer to the array of Particle structures.
 * @param colors Pointer to the array of Color structures in unified memory.
 * @param numParticles Total number of particles.
 * @param maxVel Maximum velocity for normalization.
 */
__global__ void calculateColors(const Particle *particles, Color *colors, const int numParticles, const float maxVel) {
    if (const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numParticles) {
        const float vx = particles[idx].velocity.x;
        const float vy = particles[idx].velocity.y;
        const float speed = sqrtf(vx * vx + vy * vy);
        colors[idx] = getVelocityColor(speed, maxVel);
    }
}

//---------------------------------------------------------------------
// Main function

/**
 * @brief Main entry point of the application.
 *
 * Creates a window using raylib, allocates and initializes particles using unified memory,
 * updates them with CUDA kernels, and draws them with computed colors.
 *
 * @return int Exit status.
 */
int main() {
    constexpr int screenWidth = 1920;
    constexpr int screenHeight = 1080;
    InitWindow(screenWidth, screenHeight, "SPH_CUDA with raylib and CUDA");

    constexpr int numParticles = 10000;
    Particle *particles = nullptr;
    Color *particleColors = nullptr;

    // Allocate unified memory for particles and their colors.
    cudaMallocManaged(&particles, numParticles * sizeof(Particle));
    cudaMallocManaged(&particleColors, numParticles * sizeof(Color));

    // Initialize particles with random positions and velocities.
    for (int i = 0; i < numParticles; ++i) {
        particles[i].position = {
            static_cast<float>(GetRandomValue(0, screenWidth)),
            static_cast<float>(GetRandomValue(0, screenHeight))
        };
        particles[i].velocity = {
            static_cast<float>(GetRandomValue(-5, 5)),
            static_cast<float>(GetRandomValue(-5, 5))
        };
        particles[i].mass = 1.0f;
    }

    constexpr int threadsPerBlock = 256;
    constexpr int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    SetTargetFPS(60);

    // Main loop: update particles and compute colors on the GPU, then draw them.
    while (!WindowShouldClose()) {
        constexpr float maxVelocity = 10.0f;
        // Update particle positions.
        updateParticles<<<blocks, threadsPerBlock>>>(particles, numParticles);
        cudaDeviceSynchronize();

        // Calculate particle colors based on their velocities.
        calculateColors<<<blocks, threadsPerBlock>>>(particles, particleColors, numParticles, maxVelocity);
        cudaDeviceSynchronize();

        BeginDrawing();
        ClearBackground(BLACK);
        // Draw each particle with its computed color.
        const double start_time = GetTime();
        for (int i = 0; i < numParticles; ++i) {
            DrawCircle(static_cast<int>(particles[i].position.x),
                       static_cast<int>(particles[i].position.y),
                       5.0f, particleColors[i]);
        }
        const double end_time = GetTime();
        std::cout << "Time taken to draw particles: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
        EndDrawing();
    }

    // Free unified memory.
    cudaFree(particles);
    cudaFree(particleColors);
    CloseWindow();
    return 0;
}
