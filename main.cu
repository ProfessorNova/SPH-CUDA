/**
 * @file main.cu
 * @brief SPH Simulation using CUDA kernels for physics and raylib for real-time visualization.
 *
 * This code implements a basic Smoothed Particle Hydrodynamics simulation.
 * It calculates particle densities using the poly6 kernel, computes pressures,
 * pressure forces (using the spiky kernel gradient), viscosity forces, applies gravity,
 * integrates the motion, and enforces boundary reflections.
 *
 * The simulation domain is defined in “simulation units” and then scaled for display.
 */

#include "raylib.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

//--------------------------------------------------------------------------------------
// Simulation Hyperparameters and Domain (simulation units)
//--------------------------------------------------------------------------------------
constexpr int N = 1000; // Number of particles
constexpr float DIM_SIZE_X = 40.0f; // Domain size in x (simulation units)
constexpr float DIM_SIZE_Y = 20.0f; // Domain size in y (simulation units)
constexpr float SCALE = 50.0f; // Scale factor: simulation units -> pixels
constexpr float H = 1.0f; // Smoothing radius
constexpr float K = 5000.0f; // Gas constant (stiffness)
constexpr float RHO0 = 3.0f; // Rest density
constexpr float MU = 10.0f; // Viscosity coefficient
constexpr float GRAVITY = 50.0f; // Gravity constant (applied in positive y direction)
constexpr float ANIMATION_FPS = 60.0f; // Animation frames per second
constexpr float DT = 0.005f; // Time step for integration
constexpr float BOUND_RADIUS = 0.05f; // Minimal allowed position from boundaries
constexpr float DAMPING = -0.5f; // Damping factor upon collision with boundaries

//--------------------------------------------------------------------------------------
// Mouse Interaction Parameters
//--------------------------------------------------------------------------------------
constexpr float MOUSE_STRENGTH = 10000.0f; // Force magnitude for mouse interaction (used as repulsion when negative)
constexpr float MOUSE_INTERACTION_RADIUS = 3.0f; // Radius (in simulation units) within which particles are affected

//--------------------------------------------------------------------------------------
// Particle Structure
//--------------------------------------------------------------------------------------
struct Particle {
    float2 position; // Position in simulation space
    float2 oldPosition; // Previous position (for more stable integration)
    float2 velocity; // Velocity
    float2 force; // Accumulated force
    float mass; // Mass
    float density; // Computed density
    float pressure; // Computed pressure
};

//--------------------------------------------------------------------------------------
// Device helper functions for float2 operations
//--------------------------------------------------------------------------------------
__device__ __host__ float lengthF2(const float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ __host__ float2 subtractF2(const float2 a, const float2 b) {
    float2 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
}

//--------------------------------------------------------------------------------------
// Kernel Functions (Device/Host)
//--------------------------------------------------------------------------------------

/**
 * @brief Poly6 kernel function for density estimation.
 */
__device__ __host__ float W_poly6(const float2 r, const float h) {
    if (const float rLen = lengthF2(r); rLen <= h) {
        return (315.0f / (64.0f * M_PI * powf(h, 9))) * powf((h * h - rLen * rLen), 3);
    }
    return 0.0f;
}

/**
 * @brief Spiky kernel derivative (scalar part) used for pressure force.
 */
__device__ __host__ float W_spiky_derivative(const float2 r, const float h) {
    if (const float rLen = lengthF2(r); rLen > 0.0f && rLen <= h) {
        return (-45.0f / (M_PI * powf(h, 6))) * powf((h - rLen), 2);
    }
    return 0.0f;
}

/**
 * @brief Spiky kernel gradient (vector) for pressure force.
 */
__device__ __host__ float2 W_spiky_grad(const float2 r, const float h) {
    const float rLen = lengthF2(r);
    float2 grad = {0.0f, 0.0f};
    if (rLen > 0.0f && rLen <= h) {
        const float factor = W_spiky_derivative(r, h) / rLen;
        grad.x = r.x * factor;
        grad.y = r.y * factor;
    }
    return grad;
}

/**
 * @brief Viscosity kernel Laplacian for viscosity force.
 */
__device__ __host__ float W_viscosity_laplacian(const float2 r, const float h) {
    if (const float rLen = lengthF2(r); rLen <= h) {
        return (45.0f / (M_PI * powf(h, 6))) * (h - rLen);
    }
    return 0.0f;
}

/**
 * @brief Compute a color from the particle's speed.
 *
 * Returns a color interpolated from blue (slow) to red (fast).
 */
__device__ __host__ Color getVelocityColor(const float speed, const float maxVel) {
    float t = speed / maxVel;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    // Piecewise interpolation:
    // BLUE (0.0) -> GREEN (0.33) -> YELLOW (0.66) -> RED (1.0)
    if (t < 0.33f) {
        const float localT = t / 0.33f;
        constexpr Color c1 = {0, 0, 255, 255};
        constexpr Color c2 = {0, 255, 0, 255};
        Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    } else if (t < 0.66f) {
        const float localT = (t - 0.33f) / 0.33f;
        constexpr Color c1 = {0, 255, 0, 255};
        constexpr Color c2 = {255, 255, 0, 255};
        Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    } else {
        const float localT = (t - 0.66f) / 0.34f;
        constexpr Color c1 = {255, 255, 0, 255};
        constexpr Color c2 = {255, 0, 0, 255};
        Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    }
}

/**
 * @brief Apply mouse interaction force to particles.
 *
 * This CUDA kernel applies a repulsive force to each particle based on the current mouse position.
 * The force is applied only if a particle is within a specified interaction radius.
 * The force magnitude is inversely proportional to the distance between the particle and the mouse,
 * and is scaled by the provided strength (which should be negative for repulsion).
 *
 * @param particles Pointer to the array of Particle structures.
 * @param N Number of particles.
 * @param mousePos Mouse position in simulation coordinates.
 * @param strength Coefficient for the interaction force.
 * @param interactionRadius Radius within which particles are affected.
 */
__global__ void applyMouseForce(Particle *particles, const int N, const float2 mousePos, const float strength,
                                const float interactionRadius) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        float2 direction;
        direction.x = mousePos.x - particles[i].position.x;
        direction.y = mousePos.y - particles[i].position.y;
        // Only apply force if within the interaction radius
        if (const float distance = lengthF2(direction); distance < interactionRadius) {
            if (distance > interactionRadius * 0.8f or strength < 0.0f) {
                constexpr float epsilon = 0.01f;
                // Normalize the direction vector
                direction.x /= distance;
                direction.y /= distance;
                // Compute force magnitude inversely proportional to the distance
                const float forceMag = strength / (distance + epsilon);
                // Since left mouse button should push particles away,
                // we reverse the direction (or use a negative strength).
                particles[i].force.x += forceMag * direction.x;
                particles[i].force.y += forceMag * direction.y;
            }
        }
    }
}

/**
 * @brief Compute density for each particle.
 * Density is calculated by summing over all particles using the poly6 kernel.
 */
__global__ void computeDensity(Particle *particles, const int N, const float h) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        float density = 0.0f;
        for (int j = 0; j < N; j++) {
            const float2 r = subtractF2(particles[i].position, particles[j].position);
            density += particles[j].mass * W_poly6(r, h);
        }
        particles[i].density = density;
    }
}

/**
 * @brief Compute pressure for each particle.
 * Pressure is computed using p = K*(density - RHO0).
 */
__global__ void computePressure(Particle *particles, const int N, const float K, const float RHO0) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        particles[i].pressure = K * (particles[i].density - RHO0);
    }
}

/**
 * @brief Compute pressure forces on each particle.
 * The force is accumulated into the particle's force field.
 */
__global__ void computePressureForces(Particle *particles, const int N, const float h) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        float2 force = {0.0f, 0.0f};
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            const float2 r = subtractF2(particles[i].position, particles[j].position);
            if (const float rLen = lengthF2(r); rLen <= h && rLen > 0.0f) {
                auto [x, y] = W_spiky_grad(r, h);
                const float term = (particles[i].pressure + particles[j].pressure) / (2.0f * particles[j].density);
                force.x += -particles[j].mass * term * x;
                force.y += -particles[j].mass * term * y;
            }
        }
        particles[i].force.x += force.x;
        particles[i].force.y += force.y;
    }
}

/**
 * @brief Compute viscosity forces on each particle.
 */
__global__ void computeViscosityForces(Particle *particles, const int N, const float h, const float MU) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        float2 force = {0.0f, 0.0f};
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            if (const float2 r = subtractF2(particles[i].position, particles[j].position); lengthF2(r) <= h) {
                const float laplacian = W_viscosity_laplacian(r, h);
                float2 diff;
                diff.x = particles[j].velocity.x - particles[i].velocity.x;
                diff.y = particles[j].velocity.y - particles[i].velocity.y;
                force.x += MU * particles[j].mass * diff.x / particles[j].density * laplacian;
                force.y += MU * particles[j].mass * diff.y / particles[j].density * laplacian;
            }
        }
        particles[i].force.x += force.x;
        particles[i].force.y += force.y;
    }
}

/**
 * @brief Apply gravity to each particle.
 * Gravity force is added in the positive y-direction scaled by the density.
 */
__global__ void applyGravity(Particle *particles, const int N, const float GRAVITY) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        particles[i].force.y += GRAVITY * particles[i].density;
    }
}

/**
 * @brief Save the current position of each particle to oldPosition.
 * This is used for more stable integration.
 */
__global__ void saveOldPosition(Particle *particles, const int N) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        particles[i].oldPosition = particles[i].position;
    }
}

/**
 * @brief Predict the new position of each particle based on its velocity.
 * This is used for more stable integration.
 */
__global__ void predictPosition(Particle *particles, const int N, const float lookAhead = 1.0f / 120.0f) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        particles[i].position.x += particles[i].velocity.x * lookAhead;
        particles[i].position.y += particles[i].velocity.y * lookAhead;
    }
}

/**
 * @brief Integrate particle motion (Euler integration).
 * Updates velocity and position based on the accumulated force.
 */
__global__ void integrate(Particle *particles, const int N, const float dt) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        // acceleration = force / density
        const float ax = particles[i].force.x / particles[i].density;
        const float ay = particles[i].force.y / particles[i].density;
        particles[i].velocity.x += ax * dt;
        particles[i].velocity.y += ay * dt;
        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        // Reset force for next step
        particles[i].force.x = 0.0f;
        particles[i].force.y = 0.0f;
    }
}

/**
 * @brief Reset the old position of each particle to the current position.
 * This is used for more stable integration.
 */
__global__ void resetToOldPosition(Particle *particles, const int N) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        particles[i].position = particles[i].oldPosition;
    }
}

/**
 * @brief Enforce boundary conditions by reflecting particles at domain boundaries.
 */
__global__ void applyBoundaryConditions(Particle *particles, const int N, const float dimX, const float dimY,
                                        const float damping, const float radius) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
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

//--------------------------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------------------------
int main() {
    // Set up the window (scaling simulation domain to pixels)
    constexpr int windowWidth = static_cast<int>(DIM_SIZE_X * SCALE);
    constexpr int windowHeight = static_cast<int>(DIM_SIZE_Y * SCALE);
    InitWindow(windowWidth, windowHeight, "SPH Simulation with CUDA");
    SetTargetFPS(ANIMATION_FPS);
    constexpr auto calculationsPerFrame = static_cast<unsigned int>(1.0f / (DT * ANIMATION_FPS));

    // Allocate unified memory for particles
    Particle *particles = nullptr;
    cudaMallocManaged(&particles, N * sizeof(Particle));

    // Initialize particles randomly within the simulation domain; initial velocity zero.
    for (int i = 0; i < N; i++) {
        const float x = (static_cast<float>(rand()) / RAND_MAX) * DIM_SIZE_X;
        const float y = (static_cast<float>(rand()) / RAND_MAX) * DIM_SIZE_Y;
        particles[i].position = {x, y};
        particles[i].oldPosition = {x, y};
        particles[i].velocity = {0.0f, 0.0f};
        particles[i].force = {0.0f, 0.0f};
        particles[i].mass = 1.0f;
        particles[i].density = 0.0f;
        particles[i].pressure = 0.0f;
    }

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Main simulation loop
    while (!WindowShouldClose()) {
        const double totalStartTime = GetTime();
        double calcTimeAvg = 0.0;
        for (int i = 0; i < calculationsPerFrame; i++) {
            const double startTime = GetTime();
            //----------------------------------------------------------------------------------
            // Physics: compute density, pressure, forces, integrate motion, and apply boundary conditions.
            //----------------------------------------------------------------------------------
            // Save the current position to oldPosition for integration
            saveOldPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // Apply gravity
            applyGravity<<<blocks, threadsPerBlock>>>(particles, N, GRAVITY);
            cudaDeviceSynchronize();

            // Predict the new position of each particle
            predictPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // Compute density for each particle
            computeDensity<<<blocks, threadsPerBlock>>>(particles, N, H);
            cudaDeviceSynchronize();

            // Compute pressure from density
            computePressure<<<blocks, threadsPerBlock>>>(particles, N, K, RHO0);
            cudaDeviceSynchronize();

            // Compute pressure forces
            computePressureForces<<<blocks, threadsPerBlock>>>(particles, N, H);
            cudaDeviceSynchronize();

            // Compute viscosity forces
            computeViscosityForces<<<blocks, threadsPerBlock>>>(particles, N, H, MU);
            cudaDeviceSynchronize();

            // Reset particle positions to the saved old positions
            resetToOldPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            //----------------------------------------------------------------------------------
            // Mouse Interaction: Apply repulsive force if the left mouse button is pressed.
            // The mouse position is converted from screen coordinates to simulation coordinates.
            // Only particles within MOUSE_INTERACTION_RADIUS are affected.
            // A negative force (repulsion) is applied.
            //----------------------------------------------------------------------------------
            if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
                auto [x, y] = GetMousePosition();
                float2 mousePos;
                mousePos.x = x / SCALE;
                mousePos.y = y / SCALE;
                // For repulsion, pass a negative strength.
                applyMouseForce<<<blocks, threadsPerBlock>>>(particles, N, mousePos, MOUSE_STRENGTH,
                                                             MOUSE_INTERACTION_RADIUS);
                cudaDeviceSynchronize();
            } else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
                auto [x, y] = GetMousePosition();
                float2 mousePos;
                mousePos.x = x / SCALE;
                mousePos.y = y / SCALE;
                // For attraction, pass a positive strength.
                applyMouseForce<<<blocks, threadsPerBlock>>>(particles, N, mousePos, -MOUSE_STRENGTH,
                                                             MOUSE_INTERACTION_RADIUS);
                cudaDeviceSynchronize();
            }

            // Integrate motion
            integrate<<<blocks, threadsPerBlock>>>(particles, N, DT);
            cudaDeviceSynchronize();

            // Enforce boundary conditions (simulation domain: [0, DIM_SIZE_X] x [0, DIM_SIZE_Y])
            applyBoundaryConditions<<<blocks, threadsPerBlock>>>(particles, N, DIM_SIZE_X, DIM_SIZE_Y, DAMPING,
                                                                 BOUND_RADIUS);
            cudaDeviceSynchronize();

            const double endTime = GetTime();
            calcTimeAvg += (endTime - startTime) / calculationsPerFrame;
        }
        // Print average calculation time
        std::cout << "Average calculation time: " << calcTimeAvg * 1000.0 << " ms" << std::endl;

        //----------------------------------------------------------------------------------
        // Render: draw each particle as a circle.
        // Convert simulation coordinates to screen coordinates by multiplying by SCALE.
        // The particle color is determined based on its velocity.
        //----------------------------------------------------------------------------------
        BeginDrawing();
        ClearBackground(BLACK);
        for (int i = 0; i < N; i++) {
            const float screenX = particles[i].position.x * SCALE;
            const float screenY = particles[i].position.y * SCALE;
            const float speed = lengthF2(particles[i].velocity);
            // Use 10.0f as a reference maximum speed for color mapping.
            const Color col = getVelocityColor(speed, 10.0f);
            DrawCircle(static_cast<int>(screenX), static_cast<int>(screenY), 0.1f * SCALE, col);
        }
        const double totalEndTime = GetTime();
        const double totalTime = totalEndTime - totalStartTime;
        DrawText(TextFormat("Time per frame: %.2f ms -> %.2f FPS MAX.", totalTime * 1000.0, 1.0 / totalTime), 10, 10,
                 20, WHITE);
        EndDrawing();
    }

    // Clean up
    cudaFree(particles);
    CloseWindow();
    return 0;
}
