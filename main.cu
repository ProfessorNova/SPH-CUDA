/**
 * @file main.cu
 * @brief SPH Simulation using CUDA kernels for physics and raylib for real-time visualization.
 *
 * This code implements a basic Smoothed Particle Hydrodynamics simulation with grid hashing.
 * Grid hashing is used to accelerate neighbor searches by partitioning the simulation space
 * into uniform grid cells. Each particle is binned into a cell and interactions are computed
 * only with particles in neighboring cells.
 */

namespace rl {
#include "raylib.h"
}

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <GL/GL.h>

//--------------------------------------------------------------------------------------
// Simulation Hyperparameters and Domain (simulation units)
//--------------------------------------------------------------------------------------
constexpr int N = 30000; // Number of particles
constexpr float DIM_SIZE_X = 200.0f; // Domain size in x (simulation units)
constexpr float DIM_SIZE_Y = 100.0f; // Domain size in y (simulation units)
constexpr float SCALE = 15.0f; // Scale factor: simulation units -> pixels
constexpr float H = 1.0f; // Smoothing radius (also used as grid cell size)
constexpr float K = 5000.0f; // Gas constant (stiffness)
constexpr float RHO0 = 3.0f; // Rest density
constexpr float MU = 10.0f; // Viscosity coefficient
constexpr float GRAVITY = 0.0f; // Gravity constant (applied in positive y direction)
constexpr float ANIMATION_FPS = 90.0f; // Animation frames per second
constexpr float DT = 0.005f; // Time step for integration
constexpr float BOUND_RADIUS = 0.05f; // Minimal allowed position from boundaries
constexpr float DAMPING = -0.5f; // Damping factor upon collision with boundaries

//--------------------------------------------------------------------------------------
// Mouse Interaction Parameters
//--------------------------------------------------------------------------------------
constexpr float MOUSE_STRENGTH = 10000.0f; // Force magnitude for mouse interaction (repulsion when negative)
constexpr float MOUSE_INTERACTION_RADIUS = 10.0f; // Radius within which particles are affected

//--------------------------------------------------------------------------------------
// Grid Parameters (for grid hashing)
//--------------------------------------------------------------------------------------
constexpr float GRID_CELL_SIZE = H; // We set the grid cell size equal to H (smoothing radius)
int GRID_WIDTH = static_cast<int>(std::ceil(DIM_SIZE_X / GRID_CELL_SIZE));
int GRID_HEIGHT = static_cast<int>(std::ceil(DIM_SIZE_Y / GRID_CELL_SIZE));
int GRID_CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;
int MAX_PARTICLES_PER_CELL = 64; // Maximum allowed particles per grid cell

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
// Vertex Structure for Drawing (position and color)
//--------------------------------------------------------------------------------------
struct Vertex {
    float2 position; // Screen-space coordinates
    float4 color; // RGBA color (components in range [0,1])
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
    float rLen = lengthF2(r);
    if (rLen <= h) {
        return (315.0f / (64.0f * PI * powf(h, 9))) * powf((h * h - rLen * rLen), 3);
    }
    return 0.0f;
}

/**
 * @brief Spiky kernel derivative (scalar part) used for pressure force.
 */
__device__ __host__ float W_spiky_derivative(const float2 r, const float h) {
    float rLen = lengthF2(r);
    if (rLen > 0.0f && rLen <= h) {
        return (-45.0f / (PI * powf(h, 6))) * powf((h - rLen), 2);
    }
    return 0.0f;
}

/**
 * @brief Spiky kernel gradient (vector) for pressure force.
 */
__device__ __host__ float2 W_spiky_grad(const float2 r, const float h) {
    float rLen = lengthF2(r);
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
    float rLen = lengthF2(r);
    if (rLen <= h) {
        return (45.0f / (PI * powf(h, 6))) * (h - rLen);
    }
    return 0.0f;
}

/**
 * @brief Compute a color from the particle's speed.
 *
 * Returns a color interpolated from blue (slow) to red (fast).
 */
__device__ __host__ rl::Color getVelocityColor(const float speed, const float maxVel) {
    float t = speed / maxVel;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    if (t < 0.33f) {
        float localT = t / 0.33f;
        constexpr rl::Color c1 = {0, 0, 255, 255};
        constexpr rl::Color c2 = {0, 255, 0, 255};
        rl::Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    } else if (t < 0.66f) {
        float localT = (t - 0.33f) / 0.33f;
        constexpr rl::Color c1 = {0, 255, 0, 255};
        constexpr rl::Color c2 = {255, 255, 0, 255};
        rl::Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    } else {
        float localT = (t - 0.66f) / 0.34f;
        constexpr rl::Color c1 = {255, 255, 0, 255};
        constexpr rl::Color c2 = {255, 0, 0, 255};
        rl::Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    }
}

/**
 * @brief CUDA kernel to convert particle data into vertex data for drawing.
 *
 * @param particles Pointer to particle array.
 * @param vertices Pointer to output vertex buffer.
 * @param N Number of particles.
 * @param scale Scale factor to convert simulation units to pixels.
 */
__global__ void updateVertexBuffer(const Particle *particles, Vertex *vertices, const int N, const float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Convert simulation position to screen coordinates
        float screenX = particles[i].position.x * scale;
        float screenY = particles[i].position.y * scale;
        vertices[i].position = make_float2(screenX, screenY);

        // Compute a simple color based on particle speed
        float speed = lengthF2(particles[i].velocity);
        float t = speed / 10.0f; // Assume 10.0f as a max reference speed
        t = fminf(fmaxf(t, 0.0f), 1.0f);
        // Interpolate from blue (slow) to red (fast)
        float r = t;
        float g = 0.0f;
        float b = 1.0f - t;
        vertices[i].color = make_float4(r, g, b, 1.0f);
    }
}

/**
 * @brief Creates a circle texture that can be used for drawing particles.
 *
 * @param radius The radius of the circle in pixels.
 * @param col The color of the circle.
 * @return RenderTexture2D The generated circle texture.
 */
rl::RenderTexture2D CreateCircleTexture(float radius, rl::Color col) {
    int diameter = static_cast<int>(radius * 2);
    rl::RenderTexture2D texture = rl::LoadRenderTexture(diameter, diameter);
    rl::BeginTextureMode(texture);
    rl::ClearBackground(rl::BLANK);
    DrawCircle(diameter / 2, diameter / 2, radius, col);
    rl::EndTextureMode();
    return texture;
}

/**
 * @brief Update the grid by assigning each particle to a grid cell using atomic operations.
 *
 * Each particle calculates its grid cell index and atomically increments the cell counter.
 * If the cell is not full, the particle's index is written into the gridCells array.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param cellSize Size of a grid cell.
 * @param gridWidth Number of cells in x.
 * @param gridHeight Number of cells in y.
 * @param gridCounters Array counting particles per cell.
 * @param gridCells Array storing particle indices for each cell.
 * @param maxParticlesPerCell Maximum particles allowed per cell.
 */
__global__ void updateGrid(const Particle *particles, const int N, const float cellSize, const int gridWidth,
                           const int gridHeight, int *gridCounters, int *gridCells, const int maxParticlesPerCell) {
    if (const int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        // Compute grid cell coordinates from particle position
        int cellX = static_cast<int>(particles[i].position.x / cellSize);
        int cellY = static_cast<int>(particles[i].position.y / cellSize);
        // Clamp cell indices to valid range
        cellX = max(0, min(cellX, gridWidth - 1));
        cellY = max(0, min(cellY, gridHeight - 1));
        int cellIndex = cellY * gridWidth + cellX;
        // Atomically increment the counter for this cell
        int index = atomicAdd(&gridCounters[cellIndex], 1);
        if (index < maxParticlesPerCell) {
            gridCells[cellIndex * maxParticlesPerCell + index] = i;
        }
    }
}

/**
 * @brief Compute density for each particle using grid hashing to limit neighbor searches.
 *
 * For each particle, only neighbors from adjacent grid cells (including its own) are examined.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param cellSize Size of a grid cell (equals smoothing length H).
 * @param gridWidth Number of cells in x.
 * @param gridHeight Number of cells in y.
 * @param gridCounters Array with particle counts per cell.
 * @param gridCells Array with particle indices per cell.
 * @param maxParticlesPerCell Maximum particles allowed per cell.
 */
__global__ void computeDensityGrid(Particle *particles, int N, float cellSize, int gridWidth, int gridHeight,
                                   const int *gridCounters, const int *gridCells, int maxParticlesPerCell) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Determine particle i's grid cell
        int cellX = static_cast<int>(particles[i].position.x / cellSize);
        int cellY = static_cast<int>(particles[i].position.y / cellSize);
        float density = 0.0f;
        // Loop over neighbor cells (3x3 block)
        for (int ny = cellY - 1; ny <= cellY + 1; ny++) {
            for (int nx = cellX - 1; nx <= cellX + 1; nx++) {
                if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight) {
                    int cellIndex = ny * gridWidth + nx;
                    int count = gridCounters[cellIndex];
                    for (int j = 0; j < count; j++) {
                        int neighborIndex = gridCells[cellIndex * maxParticlesPerCell + j];
                        float2 r = subtractF2(particles[i].position, particles[neighborIndex].position);
                        float rLen = lengthF2(r);
                        if (rLen <= cellSize) {
                            density += particles[neighborIndex].mass * W_poly6(r, cellSize);
                        }
                    }
                }
            }
        }
        particles[i].density = density;
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
            if (distance > interactionRadius * 0.5f or strength < 0.0f) {
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
 * @brief Compute pressure for each particle.
 * Pressure is computed using p = K*(density - RHO0).
 */
__global__ void computePressure(Particle *particles, const int N, const float K, const float RHO0) {
    if (const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N) {
        particles[i].pressure = K * (particles[i].density - RHO0);
    }
}

/**
 * @brief Compute pressure forces for each particle using grid hashing.
 *
 * For each particle, only neighbors from adjacent grid cells are examined to compute the spiky gradient.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param cellSize Size of a grid cell (equals H).
 * @param gridWidth Number of cells in x.
 * @param gridHeight Number of cells in y.
 * @param gridCounters Array with particle counts per cell.
 * @param gridCells Array with particle indices per cell.
 * @param maxParticlesPerCell Maximum particles allowed per cell.
 */
__global__ void computePressureForcesGrid(Particle *particles, int N, float cellSize, int gridWidth, int gridHeight,
                                          const int *gridCounters, const int *gridCells, int maxParticlesPerCell) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 force = {0.0f, 0.0f};
        int cellX = static_cast<int>(particles[i].position.x / cellSize);
        int cellY = static_cast<int>(particles[i].position.y / cellSize);
        for (int ny = cellY - 1; ny <= cellY + 1; ny++) {
            for (int nx = cellX - 1; nx <= cellX + 1; nx++) {
                if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight) {
                    int cellIndex = ny * gridWidth + nx;
                    int count = gridCounters[cellIndex];
                    for (int j = 0; j < count; j++) {
                        int neighborIndex = gridCells[cellIndex * maxParticlesPerCell + j];
                        if (neighborIndex == i) continue;
                        float2 r = subtractF2(particles[i].position, particles[neighborIndex].position);
                        float rLen = lengthF2(r);
                        if (rLen <= cellSize && rLen > 0.0f) {
                            float2 grad = W_spiky_grad(r, cellSize);
                            float term = (particles[i].pressure + particles[neighborIndex].pressure) /
                                         (2.0f * particles[neighborIndex].density);
                            force.x += -particles[neighborIndex].mass * term * grad.x;
                            force.y += -particles[neighborIndex].mass * term * grad.y;
                        }
                    }
                }
            }
        }
        particles[i].force.x += force.x;
        particles[i].force.y += force.y;
    }
}

/**
 * @brief Compute viscosity forces for each particle using grid hashing.
 *
 * For each particle, only neighbors from adjacent grid cells are examined.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param cellSize Size of a grid cell (equals H).
 * @param gridWidth Number of cells in x.
 * @param gridHeight Number of cells in y.
 * @param gridCounters Array with particle counts per cell.
 * @param gridCells Array with particle indices per cell.
 * @param maxParticlesPerCell Maximum particles allowed per cell.
 * @param MU Viscosity coefficient.
 */
__global__ void computeViscosityForcesGrid(Particle *particles, const int N, const float cellSize, const int gridWidth,
                                           const int gridHeight, const int *gridCounters, const int *gridCells,
                                           const int maxParticlesPerCell, const float MU) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 force = {0.0f, 0.0f};
        int cellX = static_cast<int>(particles[i].position.x / cellSize);
        int cellY = static_cast<int>(particles[i].position.y / cellSize);
        for (int ny = cellY - 1; ny <= cellY + 1; ny++) {
            for (int nx = cellX - 1; nx <= cellX + 1; nx++) {
                if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight) {
                    int cellIndex = ny * gridWidth + nx;
                    int count = gridCounters[cellIndex];
                    for (int j = 0; j < count; j++) {
                        int neighborIndex = gridCells[cellIndex * maxParticlesPerCell + j];
                        if (neighborIndex == i) continue;
                        float2 r = subtractF2(particles[i].position, particles[neighborIndex].position);
                        if (lengthF2(r) <= cellSize) {
                            float laplacian = W_viscosity_laplacian(r, cellSize);
                            float2 diff;
                            diff.x = particles[neighborIndex].velocity.x - particles[i].velocity.x;
                            diff.y = particles[neighborIndex].velocity.y - particles[i].velocity.y;
                            force.x += MU * particles[neighborIndex].mass * diff.x / particles[neighborIndex].density *
                                    laplacian;
                            force.y += MU * particles[neighborIndex].mass * diff.y / particles[neighborIndex].density *
                                    laplacian;
                        }
                    }
                }
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
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 */
__global__ void saveOldPosition(Particle *particles, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].oldPosition = particles[i].position;
    }
}

/**
 * @brief Predict the new position of each particle based on its velocity.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param lookAhead Time factor for prediction.
 */
__global__ void predictPosition(Particle *particles, const int N, const float lookAhead = 1.0f / 120.0f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].position.x += particles[i].velocity.x * lookAhead;
        particles[i].position.y += particles[i].velocity.y * lookAhead;
    }
}

/**
 * @brief Integrate particle motion (Euler integration).
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param dt Time step.
 */
__global__ void integrate(Particle *particles, const int N, const float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
 * @brief Reset the current position to the saved old position.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 */
__global__ void resetToOldPosition(Particle *particles, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].position = particles[i].oldPosition;
    }
}

/**
 * @brief Apply boundary conditions by reflecting particles at the domain boundaries.
 *
 * @param particles Pointer to particle array.
 * @param N Number of particles.
 * @param dimX Domain size in x.
 * @param dimY Domain size in y.
 * @param damping Damping factor upon collision.
 * @param radius Minimal allowed distance from boundaries.
 */
__global__ void applyBoundaryConditions(Particle *particles, const int N, const float dimX, const float dimY,
                                        const float damping, const float radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

//--------------------------------------------------------------------------------------
// Main function
//--------------------------------------------------------------------------------------
/**
 * @brief Main function for the SPH simulation.
 *
 * Sets up the simulation window, initializes particles, allocates grid arrays,
 * and enters the main loop where physics and rendering are performed.
 *
 * @return int Exit code.
 */
int main() {
    // Set up the window (scaling simulation domain to pixels)
    constexpr int windowWidth = static_cast<int>(DIM_SIZE_X * SCALE);
    constexpr int windowHeight = static_cast<int>(DIM_SIZE_Y * SCALE);
    rl::InitWindow(windowWidth, windowHeight, "SPH Simulation with CUDA - Grid Hashing");
    rl::SetTargetFPS(ANIMATION_FPS);
    constexpr auto calculationsPerFrame = static_cast<unsigned int>(1.0f / (DT * ANIMATION_FPS));

    // Allocate unified memory for particles
    Particle *particles = nullptr;
    cudaMallocManaged(&particles, N * sizeof(Particle));

    // Allocate unified memory for grid arrays
    int *gridCounters = nullptr;
    cudaMallocManaged(&gridCounters, GRID_CELL_COUNT * sizeof(int));
    int *gridCells = nullptr;
    cudaMallocManaged(&gridCells, GRID_CELL_COUNT * MAX_PARTICLES_PER_CELL * sizeof(int));

    // Initialize particles randomly within the simulation domain; initial velocity zero.
    for (int i = 0; i < N; i++) {
        float x = (static_cast<float>(rand()) / RAND_MAX) * DIM_SIZE_X;
        float y = (static_cast<float>(rand()) / RAND_MAX) * DIM_SIZE_Y;
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

    // Create the pre-rendered particle texture.
    rl::RenderTexture2D particleTexture = CreateCircleTexture(0.1f * SCALE, rl::WHITE);

    // Main simulation loop
    while (!rl::WindowShouldClose()) {
        double calcTimeAvg = 0.0;

        // Run simulation updates several times per frame
        for (int i = 0; i < calculationsPerFrame; i++) {
            const double startTime = rl::GetTime();

            // Save current positions for stable integration
            saveOldPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // Apply gravity
            applyGravity<<<blocks, threadsPerBlock>>>(particles, N, GRAVITY);
            cudaDeviceSynchronize();

            // Apply gravity
            // (Gravity is applied before computing density)
            // Each particle's force is incremented by GRAVITY scaled by its density later.
            // Here, we call the kernel to add gravity.
            applyBoundaryConditions<<<blocks, threadsPerBlock>>>(particles, N, DIM_SIZE_X, DIM_SIZE_Y, DAMPING,
                                                                 BOUND_RADIUS);
            cudaDeviceSynchronize();

            // Predict new positions based on current velocities
            predictPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // === Grid Hashing: Reset grid counters ===
            cudaMemset(gridCounters, 0, GRID_CELL_COUNT * sizeof(int));
            // Build grid: assign each particle to a grid cell.
            updateGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT,
                                                    gridCounters, gridCells, MAX_PARTICLES_PER_CELL);
            cudaDeviceSynchronize();

            // Compute density using grid-based neighbor search
            computeDensityGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT,
                                                            gridCounters, gridCells, MAX_PARTICLES_PER_CELL);
            cudaDeviceSynchronize();

            // Compute pressure based on density
            computePressure<<<blocks, threadsPerBlock>>>(particles, N, K, RHO0);
            cudaDeviceSynchronize();

            // Compute pressure forces using grid-based neighbor search
            computePressureForcesGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH,
                                                                   GRID_HEIGHT,
                                                                   gridCounters, gridCells, MAX_PARTICLES_PER_CELL);
            cudaDeviceSynchronize();

            // Compute viscosity forces using grid-based neighbor search
            computeViscosityForcesGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH,
                                                                    GRID_HEIGHT,
                                                                    gridCounters, gridCells, MAX_PARTICLES_PER_CELL,
                                                                    MU);
            cudaDeviceSynchronize();

            // Reset particle positions to the saved old positions (for more stable integration)
            resetToOldPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // Mouse interaction forces
            if (IsMouseButtonDown(rl::MOUSE_LEFT_BUTTON)) {
                auto [x, y] = rl::GetMousePosition();
                float2 mousePos = {x / SCALE, y / SCALE};
                // For repulsion, pass a negative strength.
                applyMouseForce<<<blocks, threadsPerBlock>>>(particles, N, mousePos, MOUSE_STRENGTH,
                                                             MOUSE_INTERACTION_RADIUS);
                cudaDeviceSynchronize();
            } else if (IsMouseButtonDown(rl::MOUSE_RIGHT_BUTTON)) {
                auto [x, y] = rl::GetMousePosition();
                float2 mousePos = {x / SCALE, y / SCALE};
                // For attraction, pass a positive strength.
                applyMouseForce<<<blocks, threadsPerBlock>>>(particles, N, mousePos, -MOUSE_STRENGTH,
                                                             MOUSE_INTERACTION_RADIUS);
                cudaDeviceSynchronize();
            }

            // Integrate particle motion (Euler integration)
            integrate<<<blocks, threadsPerBlock>>>(particles, N, DT);
            cudaDeviceSynchronize();

            // Enforce boundary conditions
            applyBoundaryConditions<<<blocks, threadsPerBlock>>>(particles, N, DIM_SIZE_X, DIM_SIZE_Y, DAMPING,
                                                                 BOUND_RADIUS);
            cudaDeviceSynchronize();

            const double endTime = rl::GetTime();
            calcTimeAvg += (endTime - startTime) / calculationsPerFrame;
        }

        // Render and display performance metrics
        rl::BeginDrawing();
        const double renderStartTime = rl::GetTime();
        rl::ClearBackground(rl::BLACK);

        for (int i = 0; i < N; i++) {
            float screenX = particles[i].position.x * SCALE;
            float screenY = particles[i].position.y * SCALE;
            float speed = lengthF2(particles[i].velocity);
            rl::Color col = getVelocityColor(speed, 10.0f);
            rl::DrawTexture(particleTexture.texture,
                            static_cast<int>(screenX - particleTexture.texture.width / 2),
                            static_cast<int>(screenY - particleTexture.texture.height / 2),
                            col);
        }

        const double renderEndTime = rl::GetTime();
        double renderTime = renderEndTime - renderStartTime;
        double totalFrameTime = calcTimeAvg + renderTime;
        double theoreticalMaxFPS = (totalFrameTime > 0.0) ? 1.0 / totalFrameTime : 0.0;

        rl::DrawText(rl::TextFormat("Simulation Step: %.2f ms", calcTimeAvg * 1000.0), 10, 10, 20, rl::WHITE);
        rl::DrawText(rl::TextFormat("Rendering: %.2f ms", renderTime * 1000.0), 10, 30, 20, rl::WHITE);
        rl::DrawText(rl::TextFormat("Theoretical Max FPS: %.2f", theoreticalMaxFPS), 10, 50, 20, rl::WHITE);
        rl::DrawText(rl::TextFormat("Target FPS: %.0f", ANIMATION_FPS), 10, 70, 20, rl::WHITE);
        rl::DrawText(rl::TextFormat("Current FPS: %d", rl::GetFPS()), 10, 90, 20, rl::WHITE);

        rl::EndDrawing();
    }

    // Clean up resources
    cudaFree(particles);
    cudaFree(gridCounters);
    cudaFree(gridCells);
    rl::CloseWindow();
    return 0;
}
