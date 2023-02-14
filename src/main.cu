#define SDL_MAIN_HANDLED

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <SDL2/SDL.h>

#include "mandelbrot.h"

__device__ Complex complex_product(Complex *a, Complex *b){
    Complex c;
    c.real = (a->real * b->real) - (a->imaginary * b->imaginary);
    c.imaginary = (a->real * b->imaginary) + (a->imaginary * b->real);

    return c;
}

/* ------------------------------------------------ */

__device__ Complex complex_square(Complex *a){
    Complex b;
    b.real = (a->real * a->real) - (a->imaginary * a->imaginary);
    b.imaginary = 2 * a->real * a->imaginary;

    return b;
}

/* ------------------------------------------------ */

__device__ float complex_module(Complex *a){
    return sqrt((a->imaginary * a->imaginary) + (a->real * a->real));
}

/* ------------------------------------------------ */

__device__ Complex complex_sum(Complex *a, Complex *b){
    Complex c;
    c.real = a->real + b->real;
    c.imaginary = a->imaginary + b->imaginary;

    return c;
}

/* ------------------------------------------------ */

__device__ Vector2d ScreenToImageCoordonate(Render_Parameter *param, Vector2d *pos){
    float width = (int)param->dimension.x;
    float height = (int)param->dimension.y;
    float zoom = param->zoom;
    float x = pos->x;
    float y = pos->y;

    Vector2d imageCoord = {0};

    float rate = width / height;

    imageCoord.x = (((x/width) - 0.5)* rate * 3 * (1/zoom)) + param->position.x;
    imageCoord.y = (((y/height) - 0.5) * 3 * (1/zoom)) + param->position.y;

    return imageCoord;
}

/* ------------------------------------------------ */

__device__ Complex CoordToComplex(Vector2d *vector){
    Complex complex;

    complex.real = vector->x;
    complex.imaginary = vector->y;

    return complex;
}

/* ------------------------------------------------ */

__device__ Complex fractal_reccurence(Complex prevZn, Vector2d *pos){
    Complex nextZn;
    Complex coord = CoordToComplex(pos);
    Complex square = complex_square(&prevZn);
    nextZn = complex_sum(&coord, &square);

    return nextZn;
}

/* ------------------------------------------------ */

__device__ char IsConverge(Vector2d *pos, int maxIt){
    Complex zn = {0, 0};

    for(int i = 0; i < maxIt; i++){
        zn = fractal_reccurence(zn, pos);
        if(complex_module(&zn) >= 2){
            return (char)i;
        }
    }

    return (char)0;
}

/* ------------------------------------------------ */

__device__ double sigmoid(float rate){
    return 1 / (1 + pow(2, -20 * rate + 10));
}

/* ------------------------------------------------ */

__global__ void calcFractal(Render_Parameter *param, char *array){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= (param->dimension.x * param->dimension.y))
        return;

    Vector2d pos;
    pos.y = (float)floor(i / param->dimension.x);
    pos.x = (float)(i % (int)param->dimension.x);

    Vector2d coord = ScreenToImageCoordonate(param, &pos);

    char maxIt = IsConverge(&coord, param->maxIt);
    double rate = (double)maxIt / (double)param->maxIt;
    double colo = sigmoid(rate);

    array[i*4] = (char)255;
    array[i*4 + 1] = rate * (double)255;
    array[i*4 + 2] = colo * (double)255;
    array[i*4 + 3] = colo * (double)255;

    // array[i*4 + 1] = maxIt;
    // array[i*4 + 2] = maxIt;
    // array[i*4 + 3] = maxIt;
}

/* ------------------------------------------------ */

Render_Parameter initRenderParam(Vector2d *dim){
    Render_Parameter window;
    window.dimension.x = dim->x;
    window.dimension.y = dim->y;
    window.zoom = 1;
    window.position.x = -0.5;
    window.position.y = 0;
    window.maxIt = 100;
    window.size = dim->x * dim->y;
    window.time = 0;

    return window;
}



/* ------------------------------------------------ */

void renderFractal(Render_Parameter *param, char *array, float *time){
    Render_Parameter *render_d = param->render_d;
    char *array_d = param->array_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(render_d, param, sizeof(Render_Parameter), cudaMemcpyHostToDevice);

    calcFractal<<<(int)(param->size / 1024) + 1, 1024>>>(render_d, array_d);

    cudaMemcpy(array, array_d, param->size * sizeof(char) * 4, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);
}

/* ------------------------------------------------ */

void printRenderParameter(Render_Parameter *param){
    printf("\n\n-----------------------------------\n");
    printf("Position: x:%f, y:%f\n", param->position.x, param->position.y);
    printf("dimension: x:%d, y:%d\n", (int)param->dimension.x, (int)param->dimension.y);
    printf("maxIt: %d\n", (int)param->maxIt);
    printf("size: %d\n", (int)param->size);
    printf("time: %f\n", param->time);
    printf("zoom: %f\n", param->zoom);
    printf("-----------------------------------\n\n");
}

/* ------------------------------------------------ */

void initWindows(SDL_Renderer **renderer, SDL_Window **window){
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "[DEBUG] > %s", SDL_GetError());
        return;
    }

    if (SDL_CreateWindowAndRenderer(1000, 1000, SDL_WINDOW_SHOWN, window, renderer) < 0)
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "[DEBUG] > %s", SDL_GetError());        
        SDL_Quit(); 
        return;
    }
    SDL_SetHint( SDL_HINT_RENDER_SCALE_QUALITY, "2" );
}

/* ------------------------------------------------ */

void mainloop(void){
    
}

/* ------------------------------------------------ */

void update_Texture(Render_Parameter *window, char *array, SDL_Texture *texture, SDL_Renderer *renderer){
    renderFractal(window, array, &window->time);
    printf("Time:  %3.1f ms \n", window->time);
    SDL_UpdateTexture(texture, NULL, array, window->dimension.x * 4);

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

int main(int argc, char* argv[]){
    const int width = 2000;
    const int height = 2000;
    Vector2d dim = {(float)width, (float)height};
    Render_Parameter window = initRenderParam(&dim);

    const int size = window.size;
    char *array = (char*)malloc(sizeof(char) * size * 4);
    
    cudaMalloc(&window.array_d, window.size * sizeof(char) * 4);
    cudaMalloc(&window.render_d, sizeof(Render_Parameter));

    SDL_Window* pWindow{ nullptr };     
    SDL_Renderer* pRenderer{ nullptr };
    SDL_Event events;
    initWindows(&pRenderer, &pWindow);

    bool isOpen{ true };

    SDL_Texture *texture = SDL_CreateTexture(pRenderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);

    

    while (isOpen)
    {

        update_Texture(&window, array, texture, pRenderer);

        while (SDL_PollEvent(&events))
        {
            switch (events.type)
            {
                case SDL_QUIT:
                    isOpen = false;
                    break;
            }
        }
    }

    cudaFree(window.array_d);
    cudaFree(window.render_d);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    SDL_Quit();

    free(array);

    return EXIT_SUCCESS;
}
