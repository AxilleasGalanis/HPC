#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

__global__ void d_histogram(int * hist_out, unsigned char * img_in, int img_size);
__global__ void d_lut(int *lut, int *hist, int img_size);
__global__ void d_histogram_result(unsigned char * img_out, unsigned char * img_in, 
                            int * lut, int img_size);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

#endif
