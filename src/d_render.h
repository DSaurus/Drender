#ifndef DEPTH_PCLOUD_RENDER_CUDA
#define DEPTH_PCLOUD_RENDER_CUDA

void depth_pcloud_render_cuda(float* pcloud, int* zbuf, int batch_size, int n, int h, int w);
void depth_pcloud_render_idx_cuda(float* pcloud, int* zbuf, int* idbuf, int batch_size, int n, int h, int w);

#endif