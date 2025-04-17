# Compile helpers.cpp to object file
g++ -c helpers.cpp -o helpers.o -I/usr/include/SDL2

# Compile and link cuda_sdl2.cu with helpers.o
nvcc -o cuda_sdl2 cuda_sdl2.cu helpers.o -I/usr/include/SDL2 -L/usr/lib -lSDL2