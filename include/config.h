/**
* @file config.h
 * @brief Contains simulation constants and hyperparameters.
 *
 * This file defines the simulation domain, particle count, time step, and other constants.
 */
#ifndef CONFIG_H
#define CONFIG_H

constexpr int N = 20000; // Number of particles
constexpr float DIM_SIZE_X = 100.0f; // Domain width in simulation units
constexpr float DIM_SIZE_Y = 50.0f; // Domain height in simulation units
constexpr float SCALE = 30.0f; // Scale factor: simulation units to pixels
constexpr float H = 1.0f; // Smoothing radius (and grid cell size)
constexpr float K = 1000.0f; // Gas constant (stiffness)
constexpr float RHO0 = 3.0f; // Rest density
constexpr float MU = 10.0f; // Viscosity coefficient
constexpr float GRAVITY = 0.0f; // Gravity constant (applied in positive y direction)
constexpr float ANIMATION_FPS = 60.0f; // Animation frames per second
constexpr float DT = 0.005f; // Time step for integration
constexpr float BOUND_RADIUS = 0.05f; // Minimal allowed position from boundaries
constexpr float DAMPING = -0.5f; // Damping factor upon collision with boundaries

// Mouse interaction parameters
constexpr float MOUSE_STRENGTH = 5000.0f; // Force magnitude for mouse interaction
constexpr float MOUSE_INTERACTION_RADIUS = 5.0f; // Radius within which particles are affected

// Grid parameters
constexpr float GRID_CELL_SIZE = H; // Grid cell size equals smoothing radius H
extern int GRID_WIDTH; // Computed based on DIM_SIZE_X and GRID_CELL_SIZE
extern int GRID_HEIGHT; // Computed based on DIM_SIZE_Y and GRID_CELL_SIZE
extern int GRID_CELL_COUNT; // GRID_WIDTH * GRID_HEIGHT
constexpr int MAX_PARTICLES_PER_CELL = 64;

#endif // CONFIG_H
