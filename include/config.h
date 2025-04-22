/**
* @file config.h
 * @brief Contains simulation constants and hyperparameters.
 *
 * This file defines the simulation domain, particle count, time step, and other constants.
 */
#ifndef CONFIG_H
#define CONFIG_H

constexpr int N = 15000;
constexpr float PARTICLE_MASS = 1.0f;
constexpr float DIM_SIZE_X = 10.0f;
constexpr float DIM_SIZE_Y = 6.0f;
constexpr float SCALE = 200.0f; // Scale factor: simulation units to pixels
constexpr float H = 0.1f;
constexpr float K = 30.0f;
constexpr float RHO0 = 1000.0f;
constexpr float MU = 0.001f;
constexpr float GRAVITY = 9.81f;
constexpr float ANIMATION_FPS = 60.0f;
constexpr float DT = 0.003f;
constexpr float BOUND_RADIUS = 0.05f;
constexpr float DAMPING = -0.5f;

// Mouse interaction parameters
constexpr float MOUSE_STRENGTH = 100000.0f;
constexpr float MOUSE_INTERACTION_RADIUS = 1.0f;

// Grid parameters
constexpr float GRID_CELL_SIZE = H;
extern int GRID_WIDTH; // Computed based on DIM_SIZE_X and GRID_CELL_SIZE
extern int GRID_HEIGHT; // Computed based on DIM_SIZE_Y and GRID_CELL_SIZE
extern int GRID_CELL_COUNT; // GRID_WIDTH * GRID_HEIGHT
constexpr int MAX_PARTICLES_PER_CELL = 64; // Approximation

#endif // CONFIG_H
