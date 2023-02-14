#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <SDL2/SDL.h>
#include <time.h>
#include <string.h>

#include "mandelbrot.h"

#define MAX_SIZE 80

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

__device__ double complex_module(Complex *a){
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
    double width = (int)param->dimension.x;
    double height = (int)param->dimension.y;
    double zoom = param->zoom;
    double x = pos->x;
    double y = pos->y;

    Vector2d imageCoord = {0};

    double rate = width / height;

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

__device__ char IsConverge(Render_Parameter *window, Vector2d *pos, int maxIt){
    Complex zn = {0, 0};

    for(int i = 0; i < maxIt; i++){
        zn = fractal_reccurence(zn, pos);
        if(complex_module(&zn) >= 2){
            if(window->renderMode == 1){
                return (char)i;
            }
            return (char)255;
        }
    }

    return (char)0;
}

/* ------------------------------------------------ */

__device__ double sigmoid(double rate){
    return 1 / (1 + pow(2, -20 * rate + 10));
}

/* ------------------------------------------------ */

__global__ void calcFractal(Render_Parameter *param, char *array){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= (param->dimension.x * param->dimension.y))
        return;

    Vector2d pos;
    pos.y = (double)floor(i / param->dimension.x);
    pos.x = (double)(i % (int)param->dimension.x);

    Vector2d coord = ScreenToImageCoordonate(param, &pos);

    char maxIt = IsConverge(param, &coord, param->maxIt);

    array[i*4] = (char)255;

    if(param->renderMode == 0){
        array[i*4 + 1] = maxIt;
        array[i*4 + 2] = maxIt;
        array[i*4 + 3] = maxIt;
    } else {
        double rate = (double)maxIt / (double)param->maxIt;
        double colo = sigmoid(rate);
        array[i*4 + 1] = rate * (double)255;
        array[i*4 + 2] = colo * (double)255;
        array[i*4 + 3] = colo * (double)255;
    }
}

/* ------------------------------------------------ */

Render_Parameter initRenderParam(Vector2d *dim){
    Render_Parameter window;
    window.dimension.x = dim->x;
    window.dimension.y = dim->y;
    window.zoom = 1;
    window.position.x = -0.5;
    window.position.y = 0;
    window.initMaxIt = 100;
    window.maxIt = window.initMaxIt;
    window.size = dim->x * dim->y;
    window.time = 0;
    window.isMoving = 0;
    window.renderMode = 0;

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
    printf("Position: z= %lf + %lfi\n", param->position.x, param->position.y);
    printf("dimension: x:%d, y:%d\n", (int)param->dimension.x, (int)param->dimension.y);
    printf("maxIt: %d\n", (int)param->maxIt);
    printf("initMaxIt: %d\n", param->initMaxIt);
    printf("time: %f ms\n", param->time);
    printf("zoom: %f\n", param->zoom);
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
    // SDL_SetHint( SDL_HINT_RENDER_SCALE_QUALITY, "2" );
}

/* ------------------------------------------------ */

void update_Texture(Render_Parameter *window, char *array, SDL_Texture *texture, SDL_Renderer *renderer){
    renderFractal(window, array, &window->time);
    // printf("Time:  %3.1f ms, log(zoom) = %f\n", window->time, log(window->zoom));
    SDL_UpdateTexture(texture, NULL, array, window->dimension.x * 4);

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

/* ------------------------------------------------ */

void mainloop(Render_Parameter *window, char *array, SDL_Texture *texture, SDL_Renderer *pRenderer){
    SDL_Event events;

    int isOpen = 1;
    while (isOpen){
        update_Texture(window, array, texture, pRenderer);
        refresh_events(window, &events, &isOpen, array);
    }
}

/* ------------------------------------------------ */

void refresh_events(Render_Parameter *window, SDL_Event *events, int *isOpen, char *array){
    while (SDL_PollEvent(events)){
        switch (events->type)
        {
            case SDL_MOUSEBUTTONDOWN:
                mouseClickDownEvents(window, events);
                break;
            case SDL_MOUSEBUTTONUP:
                mouseClickUpEvents(window, events);
                break;
            case SDL_MOUSEMOTION:
                mouseMotionEvent(window, events);
                break;
            case SDL_MOUSEWHEEL:
                mouseWheelEvent(window, events);
                break;
            case SDL_KEYDOWN:
                keyDownEvents(window, events, array);
                break;
            case SDL_QUIT:
                *isOpen = 0;
                break;
        }
    }
}

/* ------------------------------------------------ */

void mouseClickDownEvents(Render_Parameter *window, SDL_Event *events){
    switch(events->button.button){
        case SDL_BUTTON_LEFT:
            window->isMoving = 1;
            break;
    }
}

/* ------------------------------------------------ */

void mouseClickUpEvents(Render_Parameter *window, SDL_Event *events){
    switch(events->button.button){
        case SDL_BUTTON_LEFT:
            window->isMoving = 0;
            break;
    }
}

/* ------------------------------------------------ */

void mouseMotionEvent(Render_Parameter *window, SDL_Event *events){
    if(window->isMoving == 1){
        window->position.x -= events->motion.xrel * 0.003 * (1/window->zoom);
        window->position.y -= events->motion.yrel * 0.003 * (1/window->zoom);
    }
}

/* ------------------------------------------------ */

void mouseWheelEvent(Render_Parameter *window, SDL_Event *events){
    float sens = 0.2;
    if(events->wheel.y < 0)
        window->zoom *= 1 - sens;
    else if(events->wheel.y > 0)
        window->zoom *= 1 + sens;
    if(window->zoom < 0.3){
        window->zoom = 0.3;
    }

    double log_zoom = log(window->zoom);

    window->maxIt = log_zoom * log_zoom * 10 + window->initMaxIt;

    if(window->maxIt < window->initMaxIt)
        window->maxIt = window->initMaxIt;
}

/* ------------------------------------------------ */

void keyDownEvents(Render_Parameter *window, SDL_Event *events, char *array){
    switch(events->key.keysym.sym){
        case SDLK_SPACE:
            if(window->renderMode == 0){
                window->renderMode = 1;
            } else {
                window->renderMode = 0;
            }
            break;
        case SDLK_RETURN:
            printRenderParameter(window);
            break;
        case SDLK_p:
            saveScreen(window, array);
            break;
    }
}

/* ------------------------------------------------ */

void saveScreen(Render_Parameter *window, char *array){
    char *picture_a = (char*)malloc(sizeof(char) * 4 * window->size);
    for(int i = 0; i < window->size; i++){
        picture_a[i*4] = array[i*4 + 1];
        picture_a[i*4 + 1] = array[i*4 + 2];
        picture_a[i*4 + 2] = array[i*4 + 3];
        picture_a[i*4 + 3] = array[i*4];
    }

    SDL_Surface *surface = SDL_CreateRGBSurfaceFrom(picture_a, window->dimension.x, window->dimension.y, 8*4, window->dimension.x * 4 * sizeof(char), 0, 0, 0, 0);

    time_t timestamp = time( NULL );
    struct tm * pTime = localtime( & timestamp );

    char file_path[ MAX_SIZE ];
    strftime( file_path, MAX_SIZE, "out/mandelbrot %d%m%Y%H%M%S.bmp", pTime);

    printf("Enregistrement de: ");
    printf("%s\n", file_path);

    if(SDL_SaveBMP(surface, file_path) != 0){
        printf("SDL_SaveBMP failed: %s\n", SDL_GetError());
    }

    SDL_FreeSurface(surface);
    free(picture_a);
}

/* ------------------------------------------------ */

#undef MAX_SIZE