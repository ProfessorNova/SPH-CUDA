# SPH-CUDA: A Smoothed Particle Hydrodynamics Simulation

This project implements a simplified Smoothed Particle Hydrodynamics (SPH) simulation using CUDA for parallel computation and raylib for real-time visualization. It uses a uniform grid to speed up neighbor searches, allowing the simulation to scale to thousands of particles.

## Result

![preview](docs/preview.gif)

The simulation showcases fluid approximating properties like water.
This simulation contains 15 000 particles and runs at consistent 60 FPS (with theoratical maximum of 90 FPS) on a RTX 3080.
It also showcases mouse interaction, where the user can push or swirl the fluid by clicking and dragging the mouse.
The different colors represent the velocity of the particles, with white being the fastest and blue being the slowest.

## Project Structure

This repository is separated into two parts:

1. The implementation of the SPH simulation using CUDA and C++ for high performance.
2. A jupyter notebook that is designed to explain the SPH algorithm and its implementation in detail.

### Directory Structure

SPH-CUDA  
├── docs  
├── include  
│ ├── config.h  
│ ├── grid.h  
│ ├── kernels.h  
│ ├── particle.h  
│ └── renderer.h  
├── python  
│ ├── particle-based_fluid.ipynb  
│ └── req.txt  
├── src  
│ ├── kernels.cu  
│ ├── main.cu  
│ └── renderer.cu  
├── .gitattributes
├── .gitignore  
├── CMakeLists.txt  
└── README.md

- **docs**: Contains images and other documentation files.
- **include**: Holds header files for various parts of the simulation.
- **python**: Contains a jupyter notebook that explains the SPH algorithm and its implementation in detail.
- **src**: Contains the CUDA and C++ source files.
- **.gitignore**: Specifies which files and folders Git should ignore.
- **.gitattributes**: Configures Git attributes for the repository.
- **CMakeLists.txt**: Build configuration for CMake.
- **README.md**: This documentation.

## Setup

This section explains how to set up the CUDA simulation as well as the jupyter notebook.

### Setup for CUDA implementation

The project containes a CMakeLists.txt file for easy building. The project is designed to be built with CMake, which will handle the.
The setup is designed for Windows, but it should work on Linux and MacOS with minor modifications.

#### Prerequisites

- A CUDA-capable GPU with the NVIDIA driver installed.
- NVIDIA CUDA Toolkit (tested with 12.8)
- CMake (3.30 or higher)
- Build tools (e.g., Visual Studio, GCC, etc.)

#### Steps

1. Clone the repository and navigate into it.
2. Create a build directory and run CMake:

   ```bash
   mkdir build && cd build
   cmake ..
   ```

3. Build the project:

   ```bash
   cmake --build .
   ```

4. Run the generated executable:

   ```bash
   .\Debug\SPH_CUDA.exe
   ```

### Setup for Jupyter Notebook

1. Go into the `python` directory.

   ```bash
   cd python
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On Linux/MacOS:

     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:

   ```bash
   pip install -r req.txt
   ```

5. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open the `particle-based_fluid.ipynb` notebook to explore the SPH algorithm and its implementation.

## Components of CUDA implementation

This section describes the main components of the CUDA simulation and how they interact.

### config.h

Defines all the simulation parameters, such as the domain size, number of particles, and time step. Keeping these
constants in a single file makes them easy to manage.

### grid.h

Contains inline helper functions for converting particle positions to grid coordinates, retrieving neighboring cell indices, and more. These functions are used by the CUDA kernels to organize particles in a grid for faster neighbor
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

Implements the rendering functions declared in renderer.h. These functions create textures and color gradients for particles and are called from the main loop to draw particles each frame.

## How the Components Interact

1. **Initialization**:

   - `main.cu` sets up the window and allocates memory.
   - Particles are randomly distributed in the simulation space.

2. **Physics**:

   - `main.cu` launches kernels from `kernels.cu` to update the grid, compute density/pressure, and integrate positions.
   - `grid.h` helper functions are used in kernels to find neighboring cells efficiently.

3. **Rendering**:

   - After the physics update, `renderer.cu` functions are called to draw each particle with a color based on its velocity.

4. **Interaction**:
   - Mouse input is processed by specific kernels in `kernels.cu` (push or swirl forces).
   - Boundary conditions prevent particles from leaving the domain.
