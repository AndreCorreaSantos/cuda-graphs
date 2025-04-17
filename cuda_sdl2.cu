#include "helpers.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>


#define WIDTH 1920
#define HEIGHT 1080





__global__ void drawNodes(unsigned char* buffer, int width, int height, Node *nodes, int numNodes, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    Node n = nodes[idx];
    n.x += 100.0f * sinf(time);

    float radius = n.value;
    float r2 = radius * radius;
    float aaWidth = 1.1f;  // anti-aliasing transition zone

    for (int u = -radius - aaWidth; u <= radius + aaWidth; u++) {
        for (int v = -radius - aaWidth; v <= radius + aaWidth; v++) {
            int x = (int)(n.x + u);
            int y = (int)(n.y + v);

            if (x < 0 || x >= width || y < 0 || y >= height) continue;

            float dist2 = u * u + v * v;
            float dist = sqrtf(dist2);
            float alpha = 0.0f;

            if (dist < radius - aaWidth) {
                alpha = 1.0f; // full opacity
            } else if (dist < radius + aaWidth) {
                // fade out
                alpha = (radius + aaWidth - dist) / (2.0f * aaWidth);
            } else {
                continue; // outside soft edge
            }

            int pi = (y * width + x) * 4;

            buffer[pi + 0] = (unsigned char)(255);
            buffer[pi + 1] = 0;
            buffer[pi + 2] = 0;
            buffer[pi + 3] = (unsigned char)(alpha * 255.0f);
        }
    }
}


// Check CUDA errors
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(-1);
    }
}

bool initCUDA(unsigned char** d_buffer, int width, int height) {
    // Allocate device buffer
    cudaError_t err = cudaMalloc(d_buffer, width * height * 4);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: cudaMalloc (" << cudaGetErrorString(err) << ")\n";
        return false;
    }
    return true;
}

void cleanupCUDA(unsigned char* d_buffer) {
    if (d_buffer) {
        checkCuda(cudaFree(d_buffer), "cudaFree");
    }
}


int main() {
    // Initialize SDL2 context
    SDL2Context sdlContext = {nullptr, nullptr, nullptr};
    if (!initSDL2(sdlContext,WIDTH,HEIGHT)) {
        return -1;
    }

    // Allocate device buffer for pixel data
    unsigned char* d_buffer = nullptr;
    checkCuda(cudaMalloc(&d_buffer, WIDTH * HEIGHT * 4), "cudaMalloc");

    // Main loop
    bool running = true;
    SDL_Event event;
    int pixelX = WIDTH / 2;
    int pixelY = HEIGHT / 2;

    int numNodes;
    Node* h_nodes = readNodes("graph.txt",&numNodes);
    Node* d_nodes;

    cudaMalloc((void**)&d_nodes,sizeof(Node)*numNodes);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Node)*numNodes,cudaMemcpyHostToDevice);

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // Clear device buffer
        checkCuda(cudaMemset(d_buffer, 0, WIDTH * HEIGHT * 4), "cudaMemset");

        // Launch CUDA kernel to draw red pixel
        float t = SDL_GetTicks() / 1000.0f;  // time in seconds as float

        int blockSize = 256; // Good default
        int numBlocks = (numNodes + blockSize - 1) / blockSize;

        drawNodes<<<numBlocks, blockSize>>>(d_buffer, WIDTH, HEIGHT, d_nodes,numNodes,t );

        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel sync");

        // Lock texture and copy CUDA buffer to texture
        void* pixels = nullptr;
        int pitch;
        if (SDL_LockTexture(sdlContext.texture, nullptr, &pixels, &pitch) != 0) {
            std::cerr << "SDL_LockTexture failed: " << SDL_GetError() << "\n";
            running = false;
        } else {
            checkCuda(cudaMemcpy(pixels, d_buffer, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost), "cudaMemcpy to texture");
            SDL_UnlockTexture(sdlContext.texture);
        }

        // Render
        SDL_RenderClear(sdlContext.renderer);
        SDL_RenderCopy(sdlContext.renderer, sdlContext.texture, nullptr, nullptr);
        SDL_RenderPresent(sdlContext.renderer);
    }

    // Cleanup
    checkCuda(cudaFree(d_buffer), "cudaFree");
    delete[] h_nodes;
    cleanupSDL2(sdlContext);
    return 0;
}