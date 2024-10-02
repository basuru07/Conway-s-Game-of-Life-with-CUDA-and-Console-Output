# Conway's Game of Life with CUDA and Console Output

This project implements Conway's Game of Life using CUDA for parallel processing and displays the result as console output. The Game of Life is a cellular automaton where cells live, die, or multiply based on a set of simple rules.

## Prerequisites

To run this project, you'll need the following:

- **CUDA 12.6**: Make sure CUDA 12.6 is installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
- **Visual Studio 2022**: This project was built using Visual Studio 2022. You can download it from [Microsoft's website](https://visualstudio.microsoft.com/).
- **CUDA SDK**: Installed along with CUDA, it contains the necessary tools and libraries to build CUDA applications.

## Setup Instructions

1. **Install CUDA 12.6**:
   - Ensure that CUDA is properly installed and added to your system's environment path.

2. **Create a CUDA Project**:
   - Open Visual Studio 2022.
   - Go to `File > New > Project` and select `CUDA 12.6 Runtime` from the available templates.
   - Set up the project with the provided source files.

3. **Clone This Repository**:
   - Clone this repository and open the project in Visual Studio 2022.

## Code Overview

The project uses CUDA to compute the next generation of the Game of Life grid in parallel. The result is displayed in the console window, where each live cell is represented by `O` and dead cells are shown as spaces.

### Main Files:
- **GameOfLife.cu**: Contains the CUDA kernels and logic to run the Game of Life simulation.

### CUDA Kernel:
The core of the simulation is the CUDA kernel `updateGrid`, which computes the next state of each cell based on its neighbors:

```cpp
__global__ void updateGrid(bool* d_grid, bool* d_newGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;

    int aliveNeighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + GRID_WIDTH) % GRID_WIDTH;
            int ny = (y + dy + GRID_HEIGHT) % GRID_HEIGHT;
            aliveNeighbors += d_grid[ny * GRID_WIDTH + nx];
        }
    }

    bool currentCell = d_grid[y * GRID_WIDTH + x];
    bool newState = (currentCell && (aliveNeighbors == 2 || aliveNeighbors == 3)) ||
        (!currentCell && aliveNeighbors == 3);
    d_newGrid[y * GRID_WIDTH + x] = newState;
}
```
## How to Run

1. Build the project in Visual Studio 2022. 
2. Run the project. The console will display the Game of Life grid, where live cells are shown as **O** and dead cells as spaces.
3. The grid updates every 100 milliseconds. You can press **Ctrl + C** to stop the simulation.
