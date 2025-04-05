/**
 * @file main.cu
 * @brief Main entry point for the SPH simulation.
 *
 * This file initializes the simulation, creates the window, and runs the main loop,
 * launching CUDA kernels for physics updates and rendering the particles.
 */

#include "config.h"
#include "particle.h"
#include "kernels.h"
#include "renderer.h"

#include <raylib.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Global grid dimensions computed from simulation settings
int GRID_WIDTH = static_cast<int>(std::ceil(DIM_SIZE_X / GRID_CELL_SIZE));
int GRID_HEIGHT = static_cast<int>(std::ceil(DIM_SIZE_Y / GRID_CELL_SIZE));
int GRID_CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;

/**
 * @brief Main function for the SPH simulation.
 *
 * @return int Exit code.
 */
int main() {
    // Initialize the simulation window (scaling simulation domain to pixels)
    constexpr int windowWidth = static_cast<int>(DIM_SIZE_X * SCALE);
    constexpr int windowHeight = static_cast<int>(DIM_SIZE_Y * SCALE);
    InitWindow(windowWidth, windowHeight, "SPH Simulation with CUDA - Grid Hashing");
    SetTargetFPS(ANIMATION_FPS);
    constexpr auto calculationsPerFrame = static_cast<unsigned int>(1.0f / (DT * ANIMATION_FPS));

    // Allocate unified memory for particles
    Particle *particles = nullptr;
    cudaMallocManaged(&particles, N * sizeof(Particle));

    // Allocate unified memory for grid arrays
    int *gridCounters = nullptr;
    cudaMallocManaged(&gridCounters, GRID_CELL_COUNT * sizeof(int));
    unsigned int *gridCells = nullptr;
    cudaMallocManaged(&gridCells, GRID_CELL_COUNT * MAX_PARTICLES_PER_CELL * sizeof(int));

    // Initialize particles with random positions within the simulation domain; initial velocity zero.
    std::random_device rd; // Seed for random number generation
    std::mt19937 mt(rd()); // Mersenne Twister random number generator
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Uniform distribution in [0, 1)
    for (int i = 0; i < N; i++) {
        float x = dist(mt) * DIM_SIZE_X;
        float y = dist(mt) * DIM_SIZE_Y;
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

    // Create pre-rendered particle texture.
    RenderTexture2D particleTexture = CreateCircleTexture(0.2f * SCALE, WHITE);

    // Main simulation loop
    while (!WindowShouldClose()) {
        double calcTimeAvg = 0.0;

        // Run simulation updates multiple times per frame
        for (unsigned int step = 0; step < calculationsPerFrame; step++) {
            double startTime = GetTime();

            // Save current positions for stable integration
            saveOldPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // Apply gravity to particles
            applyGravity<<<blocks, threadsPerBlock>>>(particles, N, GRAVITY);
            cudaDeviceSynchronize();

            // Apply boundary conditions
            applyBoundaryConditions<<<blocks, threadsPerBlock>>>(particles, N, DIM_SIZE_X, DIM_SIZE_Y, DAMPING,
                                                                 BOUND_RADIUS);
            cudaDeviceSynchronize();

            // Predict new positions based on current velocities
            predictPosition<<<blocks, threadsPerBlock>>>(particles, N, 1.0f / 120.0f);
            cudaDeviceSynchronize();

            // Reset grid counters
            cudaMemset(gridCounters, 0, GRID_CELL_COUNT * sizeof(int));
            // Update grid: assign each particle to a grid cell
            updateGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT,
                                                    gridCounters, gridCells, MAX_PARTICLES_PER_CELL);
            cudaDeviceSynchronize();

            // Compute density using grid-based neighbor search
            computeDensityGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT,
                                                            gridCounters, gridCells, MAX_PARTICLES_PER_CELL);
            cudaDeviceSynchronize();

            // Compute pressure from density
            computePressure<<<blocks, threadsPerBlock>>>(particles, N, K, RHO0);
            cudaDeviceSynchronize();

            // Compute pressure forces using grid-based neighbor search
            computePressureForcesGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH,
                                                                   GRID_HEIGHT, gridCounters, gridCells,
                                                                   MAX_PARTICLES_PER_CELL);
            cudaDeviceSynchronize();

            // Compute viscosity forces using grid-based neighbor search
            computeViscosityForcesGrid<<<blocks, threadsPerBlock>>>(particles, N, GRID_CELL_SIZE, GRID_WIDTH,
                                                                    GRID_HEIGHT, gridCounters, gridCells,
                                                                    MAX_PARTICLES_PER_CELL, MU);
            cudaDeviceSynchronize();

            // Reset particle positions to saved positions for stability
            resetToOldPosition<<<blocks, threadsPerBlock>>>(particles, N);
            cudaDeviceSynchronize();

            // Handle mouse interactions: push or swirl force
            if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
                Vector2 mousePosRaylib = GetMousePosition();
                float2 mousePos = {mousePosRaylib.x / SCALE, mousePosRaylib.y / SCALE};
                applyPushForce<<<blocks, threadsPerBlock>>>(particles, N, mousePos, -MOUSE_STRENGTH,
                                                            MOUSE_INTERACTION_RADIUS);
                cudaDeviceSynchronize();
            } else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
                Vector2 mousePosRaylib = GetMousePosition();
                float2 mousePos = {mousePosRaylib.x / SCALE, mousePosRaylib.y / SCALE};
                applySwirlForce<<<blocks, threadsPerBlock>>>(particles, N, mousePos, MOUSE_STRENGTH,
                                                             MOUSE_INTERACTION_RADIUS);
                cudaDeviceSynchronize();
            }

            // Integrate particle motion (Euler integration)
            integrate<<<blocks, threadsPerBlock>>>(particles, N, DT);
            cudaDeviceSynchronize();

            // Enforce boundary conditions again after integration
            applyBoundaryConditions<<<blocks, threadsPerBlock>>>(particles, N, DIM_SIZE_X, DIM_SIZE_Y, DAMPING,
                                                                 BOUND_RADIUS);
            cudaDeviceSynchronize();

            double endTime = GetTime();
            calcTimeAvg += (endTime - startTime) / calculationsPerFrame;
        }

        // Rendering
        BeginDrawing();
        double renderStartTime = GetTime();
        ClearBackground(BLACK);

        // Draw each particle
        for (int i = 0; i < N; i++) {
            int screenX = static_cast<int>(particles[i].position.x * SCALE);
            int screenY = static_cast<int>(particles[i].position.y * SCALE);
            float speed = lengthF2(particles[i].velocity);
            Color col = getVelocityColor(speed, 25.0f);
            DrawTexture(particleTexture.texture,
                        screenX - particleTexture.texture.width / 2,
                        screenY - particleTexture.texture.height / 2,
                        col);
        }

        double renderEndTime = GetTime();
        double renderTime = renderEndTime - renderStartTime;
        double totalFrameTime = calcTimeAvg + renderTime;
        double theoreticalMaxFPS = (totalFrameTime > 0.0) ? 1.0 / totalFrameTime : 0.0;

        DrawText(TextFormat("Simulation Step: %.2f ms", calcTimeAvg * 1000.0), 10, 10, 20, WHITE);
        DrawText(TextFormat("Rendering: %.2f ms", renderTime * 1000.0), 10, 30, 20, WHITE);
        DrawText(TextFormat("Theoretical Max FPS: %.2f", theoreticalMaxFPS), 10, 50, 20, WHITE);
        DrawText(TextFormat("Target FPS: %.0f", ANIMATION_FPS), 10, 70, 20, WHITE);
        DrawText(TextFormat("Current FPS: %d", GetFPS()), 10, 90, 20, WHITE);
        DrawText("Controls: LMB to push, RMB to swirl", 10, 130, 20, WHITE);

        EndDrawing();
    }

    // Clean up resources
    cudaFree(particles);
    cudaFree(gridCounters);
    cudaFree(gridCells);
    CloseWindow();
    return 0;
}
