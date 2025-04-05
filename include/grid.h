/**
 * @file grid.h
 * @brief Declarations of grid hashing helper functions.
 *
 * This file contains inline helper functions for grid-related operations such as computing
 * grid cell coordinates, converting positions to cell indices, and retrieving the indices
 * of neighboring grid cells. Using these helpers reduces code duplication in your kernels.
 */
#ifndef GRID_H
#define GRID_H

// Define HOST_DEVICE macro for functions callable on both host and device.
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

/**
 * @brief Compute the grid cell coordinates for a given position.
 *
 * This helper function converts a position in simulation space into grid cell coordinates.
 * It clamps the coordinates to ensure they are within the valid grid range.
 *
 * @param pos Position in simulation space.
 * @param cellSize Size of each grid cell.
 * @param gridWidth Total number of cells in the x-direction.
 * @param gridHeight Total number of cells in the y-direction.
 * @return int2 The clamped grid cell coordinates (x, y).
 */
HOST_DEVICE inline int2 getCellCoordinates(const float2 pos, float cellSize, int gridWidth, int gridHeight) {
    int cellX = static_cast<int>(pos.x / cellSize);
    int cellY = static_cast<int>(pos.y / cellSize);
    cellX = max(0, min(cellX, gridWidth - 1));
    cellY = max(0, min(cellY, gridHeight - 1));
    return make_int2(cellX, cellY);
}

/**
 * @brief Compute the grid cell index for a given position.
 *
 * This helper converts a particle's position to a one-dimensional grid cell index.
 *
 * @param pos Position in simulation space.
 * @param cellSize Size of each grid cell.
 * @param gridWidth Total number of cells in the x-direction.
 * @param gridHeight Total number of cells in the y-direction.
 * @return int The computed grid cell index.
 */
HOST_DEVICE inline int getCellIndex(const float2 pos, float cellSize, int gridWidth, int gridHeight) {
    int2 coords = getCellCoordinates(pos, cellSize, gridWidth, gridHeight);
    return coords.y * gridWidth + coords.x;
}

/**
 * @brief Retrieve the neighbor grid cell indices for a given cell.
 *
 * This helper function fills the provided array with the cell indices for the 3x3 neighborhood
 * (including the cell itself) around a given grid cell. It returns the number of valid neighbors.
 *
 * @param cellX The x coordinate of the cell.
 * @param cellY The y coordinate of the cell.
 * @param gridWidth Total number of cells in the x-direction.
 * @param gridHeight Total number of cells in the y-direction.
 * @param neighbors Output array (size must be at least 9) to store neighbor cell indices.
 * @return int The number of neighbor cell indices stored.
 */
HOST_DEVICE inline int getNeighborCellIndices(int cellX, int cellY, int gridWidth, int gridHeight, int *neighbors) {
    int count = 0;
    int startX = max(0, cellX - 1);
    int endX = min(gridWidth - 1, cellX + 1);
    int startY = max(0, cellY - 1);
    int endY = min(gridHeight - 1, cellY + 1);
    for (int y = startY; y <= endY; y++) {
        for (int x = startX; x <= endX; x++) {
            neighbors[count++] = y * gridWidth + x;
        }
    }
    return count;
}

#endif // GRID_H
