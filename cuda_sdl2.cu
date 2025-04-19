#include "helpers.h"
#include <SDL2/SDL_events.h>
#include <cmath>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>


#define WIDTH 1920
#define HEIGHT 1080





__global__ void drawNodes(unsigned char* buffer, int width, int height, Node *nodes, int numNodes, float time, PlayerData* pData) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    Node n = nodes[idx];
    n.x += sinf(time*(10.0+ 50.0/n.value)/10.0f);
    n.y += cosf(time*(10.0+ 50.0/n.value)/10.0f);
    nodes[idx] = n;

    float radius = n.value*pData->zoom;
    float r2 = radius * radius;
    float aaWidth = 1.0f;  // anti-aliasing transition zone

    for (int u = -radius - aaWidth; u <= radius + aaWidth; u++) {
        for (int v = -radius - aaWidth; v <= radius + aaWidth; v++) {
            int x = (int)(n.x*pData->zoom + u + pData->px);
            int y = (int)(n.y*pData->zoom + v + pData->py);

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
            unsigned char a = (alpha * 255.0f);
            buffer[pi + 0] = (unsigned char)(255);
            buffer[pi + 1] = 0;
            buffer[pi + 2] = 0;
            buffer[pi + 3] = (unsigned char) a;
        }
    }
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}



__device__ float clamp(float a, float b, float c){
    return min(max(a,b),c);
}
__global__ void drawEdges(unsigned char* buffer, int width, int height, Edge* edges, Node* nodes, int numEdges, PlayerData *pData)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;

    Edge e = edges[idx];
    Node n1 = nodes[e.n1];
    Node n2 = nodes[e.n2];

    int x0 = static_cast<int>(n1.x * pData->zoom + pData->px);
    int y0 = static_cast<int>(n1.y * pData->zoom + pData->py);
    int x1 = static_cast<int>(n2.x * pData->zoom + pData->px);
    int y1 = static_cast<int>(n2.y * pData->zoom + pData->py);

    int thickness = static_cast<int>((e.strength * 4.0f) + 2)*pData->zoom;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    float dist = sqrtf((dx * dx) + (dy * dy));
    if (dist > 500.0f) return;
    while (true)
    {
        for (int ox = -thickness / 2; ox <= thickness / 2; ++ox)
        {
            for (int oy = -thickness / 2; oy <= thickness / 2; ++oy)
            {
                int px = x0 + ox;
                int py = y0 + oy;

                if (px >= 0 && px < width && py >= 0 && py < height)
                {
                    int i = (py * width + px) * 4;

                    buffer[i + 0] = 0;
                    buffer[i + 1] = 0;
                    buffer[i + 2] = (unsigned char) clamp((50000.0f / dist),0.0f,255.0f);
                    buffer[i + 3] = 0;
                }
            }
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}



void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(-1);
    }
}

bool initCUDA(unsigned char** d_buffer, int width, int height) {
    // allocate device buffer
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
    // init SDL2 context
    SDL2Context sdlContext = {nullptr, nullptr, nullptr};
    if (!initSDL2(sdlContext,WIDTH,HEIGHT)) {
        return -1;
    }

    // Allocate device buffer for pixel data
    unsigned char* d_buffer = nullptr;
    checkCuda(cudaMalloc(&d_buffer, WIDTH * HEIGHT * 4), "cudaMalloc");

    // main loop
    bool running = true;
    bool mouseDown = false;
    SDL_Event event;
    int pixelX = WIDTH / 2;
    int pixelY = HEIGHT / 2;

    int numNodes;
    Node* h_nodes = readNodes("nodes.txt",&numNodes);
    Node* d_nodes;
    cudaMalloc((void**)&d_nodes,sizeof(Node)*numNodes);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Node)*numNodes,cudaMemcpyHostToDevice);


    int numEdges;
    Edge* h_edges = readEdges("edges.txt",&numEdges);
    Edge* d_edges;
    cudaMalloc((void**)&d_edges,sizeof(Node)*numEdges);
    cudaMemcpy(d_edges, h_edges, sizeof(Node)*numEdges,cudaMemcpyHostToDevice);

    PlayerData pData = {0,0, 1.0}; 

    PlayerData *h_pData = &pData;
    PlayerData *d_pData;

    cudaMalloc((void**) &d_pData,sizeof(PlayerData)); // maybe will have to move this logic to inside the game loop, so that d_pdata gets updated? or does it get set auto?
    cudaMemcpy(d_pData,h_pData,sizeof(PlayerData),cudaMemcpyHostToDevice);

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                mouseDown = true;
            }
            if (event.type == SDL_MOUSEBUTTONUP)
            {
                mouseDown = false;
            }
            if (event.type == SDL_MOUSEMOTION && mouseDown) {
                pData.px += event.motion.xrel;
                pData.py += event.motion.yrel;
            }
            if (event.type == SDL_MOUSEWHEEL) {
                float sensitivity = 0.1f;
                float oldZoom = pData.zoom;
                
                // Multiplicative zoom (better than additive)
                pData.zoom *= (1.0f + event.wheel.y * sensitivity);
                
                pData.zoom = fmax(0.1f, fmin(10.0f, pData.zoom));
                
                // Adjust pan to zoom from screen center
                pData.px = (WIDTH/2.0f) - ((WIDTH/2.0f - pData.px) * pData.zoom / oldZoom);
                pData.py = (HEIGHT/2.0f) - ((HEIGHT/2.0f - pData.py) * pData.zoom / oldZoom);
            }
            if (event.type == SDL_QUIT) {
                running = false;
            }

        }
        cudaMemcpy(d_pData, &pData, sizeof(PlayerData), cudaMemcpyHostToDevice); // updating player data after events
        // Clear device buffer
        checkCuda(cudaMemset(d_buffer, 0, WIDTH * HEIGHT * 4), "cudaMemset");

        float t = SDL_GetTicks() / 1000.0f;  // time in seconds as float
        // float t = 1.0;
        int blockSize = 256; // Good default
        int numBlocks = (numNodes + blockSize - 1) / blockSize;

        

        numBlocks = (numEdges + blockSize - 1 ) / blockSize;
        drawEdges<<<numBlocks,blockSize>>>(d_buffer,WIDTH,HEIGHT,d_edges,d_nodes,numEdges,d_pData);

        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(),"kernel sync");
        
        drawNodes<<<numBlocks, blockSize>>>(d_buffer, WIDTH, HEIGHT, d_nodes,numNodes,t ,d_pData);

        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel sync");
        // lock texture and copy CUDA buffer to texture
        void* pixels = nullptr;
        int pitch;
        if (SDL_LockTexture(sdlContext.texture, nullptr, &pixels, &pitch) != 0) {
            std::cerr << "SDL_LockTexture failed: " << SDL_GetError() << "\n";
            running = false;
        } else {
            checkCuda(cudaMemcpy(pixels, d_buffer, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost), "cudaMemcpy to texture");
            SDL_UnlockTexture(sdlContext.texture);
        }

        // sdl render
        SDL_RenderClear(sdlContext.renderer);
        SDL_RenderCopy(sdlContext.renderer, sdlContext.texture, nullptr, nullptr);
        SDL_RenderPresent(sdlContext.renderer);
    }

    // cleanup
    checkCuda(cudaFree(d_buffer), "cudaFree");
    checkCuda(cudaFree(d_edges),"cudaFree");
    delete[] h_nodes;
    delete[] h_edges;
    cleanupSDL2(sdlContext);
    return 0;
}