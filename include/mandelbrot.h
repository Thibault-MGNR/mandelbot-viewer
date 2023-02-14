#ifndef __MANDELBROT__H__
#define __MANDELBROT__H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <SDL2/SDL.h>

typedef struct Vector2d {
    float x;
    float y;
} Vector2d;

typedef struct Complex {
    float real;
    float imaginary;
} Complex;


typedef struct Render_Parameter {
    Vector2d position;
    Vector2d dimension;
    float zoom;
    int maxIt;
    int size;
    float time;
    struct Render_Parameter *render_d;
    char *array_d;
} Render_Parameter;

__device__ Complex complex_product(Complex *a, Complex *b);

__device__ Complex complex_square(Complex *a);

__device__ float complex_module(Complex *a);

__device__ Complex complex_sum(Complex *a, Complex *b);

__device__ Vector2d ScreenToImageCoordonate(Render_Parameter *param, Vector2d *pos);

__device__ Complex fractal_reccurence(Complex prevZn, Vector2d *pos);

__device__ Complex CoordToComplex(Vector2d *vector);

__device__ char IsConverge(Vector2d *pos, int maxIt);

__device__ double sigmoid(float rate);

__global__ void calcFractal(Render_Parameter *param, char *array);

void renderFractal(Render_Parameter *param, char *array, float *time);

Render_Parameter initRenderParam(Vector2d *dim);

void printRenderParameter(Render_Parameter *param);

void initWindows(SDL_Renderer **renderer, SDL_Window **window);

void mainloop(void);

#endif