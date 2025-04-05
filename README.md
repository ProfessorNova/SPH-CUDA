# SPH-CUDA: A Smoothed Particle Hydrodynamics Simulation

This project implements a simplified Smoothed Particle Hydrodynamics (SPH) simulation using CUDA for parallel
computation and raylib for real-time visualization. It uses a uniform grid to speed up neighbor searches, allowing the
simulation to scale to thousands of particles.

## Table of Contents

- Overview
- Project Structure
- Components
    - config.h
    - grid.h
    - kernels.h
    - particle.h
    - renderer.h
    - main.cu
    - kernels.cu
    - renderer.cu
- How the Components Interact
- Build and Run Instructions
- Future Improvements

## Overview

SPH is a method for simulating fluid by modeling it as a set of particles. Each particle computes its density, pressure,
and forces based on nearby particles. Using CUDA accelerates these computations by distributing them across the GPU.
raylib handles the graphics, drawing particles at positions updated each frame.

## Project Structure

SPH-CUDA  
├── include  
│ ├── config.h  
│ ├── grid.h  
│ ├── kernels.h  
│ ├── particle.h  
│ └── renderer.h  
├── src  
│ ├── kernels.cu  
│ ├── main.cu  
│ └── renderer.cu  
├── .gitignore  
├── CMakeLists.txt  
└── README.md

- **include**: Holds header files for various parts of the simulation.
- **src**: Contains the CUDA and C++ source files.
- **.gitignore**: Specifies which files and folders Git should ignore.
- **CMakeLists.txt**: Build configuration for CMake.
- **README.md**: This documentation.

## Components

### config.h

Defines all the simulation parameters, such as the domain size, number of particles, and time step. Keeping these
constants in a single file makes them easy to manage.

### grid.h

Contains inline helper functions for converting particle positions to grid coordinates, retrieving neighboring cell
indices, and more. These functions are used by the CUDA kernels to organize particles in a grid for faster neighbor
searches.

### kernels.h

Declares all CUDA kernels for:

- Updating grid structures
- Computing particle density, pressure, and forces
- Integrating particle positions
- Handling mouse interactions and boundaries

### particle.h

Defines the `Particle` struct. Each particle stores:

- Position
- Old position (for stable integration)
- Velocity
- Force
- Mass
- Density
- Pressure

Also contains inline vector math utilities (`lengthF2`, `subtractF2`) used by both host code and CUDA kernels.

### renderer.h

Declares functions for rendering particles using raylib. This includes:

- Creating textures (e.g., circles for particles)
- Generating color based on particle velocity

### main.cu

The entry point of the simulation. It:

1. Initializes the window (via raylib)
2. Allocates memory for particles and grid data
3. Randomizes initial particle positions
4. Runs the main loop:
    - Updates physics by launching CUDA kernels
    - Renders the particles with raylib
5. Cleans up and closes the window on exit

### kernels.cu

Implements the CUDA kernels declared in kernels.h. These kernels run in parallel on the GPU to handle:

- Grid updates and neighbor searches
- Density and pressure calculations
- Force accumulation (pressure, viscosity, mouse interaction, gravity)
- Integration of particle motion
- Boundary checks

### renderer.cu

Implements the rendering functions declared in renderer.h. These functions create textures and color gradients for
particles and are called from the main loop to draw particles each frame.

## How the Components Interact

1. **Initialization**:
    - `main.cu` sets up the window and allocates memory.
    - Particles are randomly distributed in the simulation space.

2. **Physics**:
    - `main.cu` launches kernels from `kernels.cu` to update the grid, compute density/pressure, and integrate
      positions.
    - `grid.h` helper functions are used in kernels to find neighboring cells efficiently.

3. **Rendering**:
    - After the physics update, `renderer.cu` functions are called to draw each particle with a color based on its
      velocity.

4. **Interaction**:
    - Mouse input is processed by specific kernels in `kernels.cu` (push or swirl forces).
    - Boundary conditions prevent particles from leaving the domain.

## Build and Run Instructions

### Prerequisites

- A CUDA-capable GPU
- CMake (3.30 or higher)
- raylib and OpenGL development libraries

### Steps

1. Clone the repository and navigate into it.
2. Create a build directory and run CMake:  
   mkdir build && cd build  
   cmake ..
3. Build the project:  
   cmake --build .
4. Run the generated executable:  
   ./SPH_CUDA

## Future Improvements

- **Performance Optimization**: Investigate better data structures or optimization techniques for neighbor searches.
- **Extended Physics**: Add surface tension or more complex boundary conditions.
- **Interactive UI**: Implement on-screen controls to change simulation parameters in real time.
- **Multi-Platform Support**: Ensure compatibility with various operating systems and compilers.

By focusing on modular design, this project demonstrates how to integrate CUDA-based physics with raylib for real-time
rendering, allowing for a straightforward approach to exploring fluid dynamics through particle-based methods.
