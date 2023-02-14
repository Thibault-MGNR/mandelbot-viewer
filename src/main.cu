#define SDL_MAIN_HANDLED

#include "mandelbrot.h"
#include "mandelbrot.cu"

int main(int argc, char* argv[]){
    const int width = 1500;
    const int height = 1500;
    Vector2d dim = {(type_of_calc)width, (type_of_calc)height};
    Render_Parameter window = initRenderParam(&dim);

    const int size = window.size;
    char *array = (char*)malloc(sizeof(char) * size * 4);
    
    cudaMalloc(&window.array_d, window.size * sizeof(char) * 4);
    cudaMalloc(&window.render_d, sizeof(Render_Parameter));

    SDL_Window* pWindow{ nullptr };     
    SDL_Renderer* pRenderer{ nullptr };
    initWindows(&pRenderer, &pWindow);

    SDL_Texture *texture = SDL_CreateTexture(pRenderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);

    mainloop(&window, array, texture, pRenderer);

    cudaFree(window.array_d);
    cudaFree(window.render_d);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    SDL_Quit();

    free(array);

    return EXIT_SUCCESS;
}

/* ------------------------------------------------ */