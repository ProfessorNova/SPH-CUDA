/**
* @file config.h
 * @brief Contains simulation constants and hyperparameters.
 *
 * This file defines the simulation domain, particle count, time step, and other constants.
 */
#ifndef CONFIG_H
#define CONFIG_H

constexpr int N = 15000; // (Number of particles)
constexpr float PARTICLE_MASS = 1.0f; // kg (fluid mass is N * PARTICLE_MASS)
constexpr float DIM_SIZE_X = 10.0f; // m (Width of the simulation domain)
constexpr float DIM_SIZE_Y = 6.0f; // m (Height of the simulation domain)
constexpr float SCALE = 100.0f; // px/m (Scale factor for rendering)
constexpr float H = 0.1f; // m (Smoothing length)
constexpr float K = 30.0f; // Pa (Gas constant)
constexpr float RHO0 = 1000.0f; // kg/m^3 (Rest density)
constexpr float MU = 0.001f; // Pa * s (Viscosity coefficient)
constexpr float GRAVITY = 9.81f; // m/s^2 (Gravity acceleration)
constexpr float ANIMATION_FPS = 60.0f; // (Frames per second for animation)
constexpr float DT = 0.003f; // s (Time step for simulation)
constexpr float BOUND_RADIUS = 0.05f; // m (Radius of the boundary around the domain)
constexpr float DAMPING = -0.5f; // (Damping factor on collision with domain walls)

// Mouse interaction parameters
constexpr float MOUSE_STRENGTH = 100000.0f; // (Force applied to particles when mouse is pressed)
constexpr float MOUSE_INTERACTION_RADIUS = 1.0f; // m (Radius of interaction with mouse)

// Grid parameters
constexpr float GRID_CELL_SIZE = H;
extern int GRID_WIDTH; // (Computed based on DIM_SIZE_X and GRID_CELL_SIZE)
extern int GRID_HEIGHT; // (Computed based on DIM_SIZE_Y and GRID_CELL_SIZE)
extern int GRID_CELL_COUNT; // (GRID_WIDTH * GRID_HEIGHT)
constexpr int MAX_PARTICLES_PER_CELL = 64; // (Approximation)

#endif // CONFIG_H
