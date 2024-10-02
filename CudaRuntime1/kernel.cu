#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <chrono>
#include <thread>

const int WINDOW_WIDTH = 80;  // Reduced width for console output
const int WINDOW_HEIGHT = 40;  // Reduced height for console output
const int CELL_SIZE = 1;  // No need for cell size, just 1x1 grid in console
const int GRID_WIDTH = WINDOW_WIDTH / CELL_SIZE;
const int GRID_HEIGHT = WINDOW_HEIGHT / CELL_SIZE;
const int FRAME_DELAY = 100;

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

void initializeGrid(bool* grid) {
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        grid[i] = rand() % 2;
    }
}

void printGrid(bool* grid) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (grid[y * GRID_WIDTH + x]) {
                printf("O");  // Alive cell
            }
            else {
                printf(" ");  // Dead cell
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    bool* h_grid = new bool[GRID_WIDTH * GRID_HEIGHT];
    bool* h_newGrid = new bool[GRID_WIDTH * GRID_HEIGHT];
    bool* d_grid;
    bool* d_newGrid;

    initializeGrid(h_grid);

    cudaMalloc(&d_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool));
    cudaMalloc(&d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool));

    cudaMemcpy(d_grid, h_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((GRID_WIDTH + blockSize.x - 1) / blockSize.x, (GRID_HEIGHT + blockSize.y - 1) / blockSize.y);

    bool quit = false;
    while (!quit) {
        updateGrid << <gridSize, blockSize >> > (d_grid, d_newGrid);
        cudaDeviceSynchronize();

        cudaMemcpy(h_grid, d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool), cudaMemcpyDeviceToHost);

        // Clear console and print grid
        system("clear");  // Use "cls" if on Windows
        printGrid(h_grid);

        std::swap(d_grid, d_newGrid);
        std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_DELAY));
    }

    cudaFree(d_grid);
    cudaFree(d_newGrid);
    delete[] h_grid;
    delete[] h_newGrid;

    return 0;
}
