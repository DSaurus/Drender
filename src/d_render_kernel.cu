#include "d_render.h"

__global__ void depth_pcloud_render_kernel(float* pcloud, int* zbuf, int batch_size, int n, int h, int w){
    int pid = blockIdx.x;
    int bid = blockIdx.y;
    float* p_data = pcloud + (bid*n*3 + pid*3);
    int x = max(0, min((int)(w*((*p_data)*0.5 + 0.5)), w-1));
    int y = max(0, min((int)(h*((*(p_data+1))*0.5 + 0.5)), h-1));
    for(int tx = max(0, x-1); tx < min(x+1, w); tx++){   
        for(int ty = max(0, y-1); ty < min(y+1, w); ty++){ 
            atomicMax(zbuf + (bid*h*w + w*y + x), (int)(*(p_data+2)*1e5));
        }
    }
}


void depth_pcloud_render_cuda(float* pcloud, int* zbuf, int batch_size, int n, int h, int w){
    // pcloud B N 3
    // zbuf B H W
    dim3 blocks_dim(n, batch_size);

    depth_pcloud_render_kernel<<<blocks_dim, 1>>>(
        pcloud, zbuf, batch_size, n, h, w
    );
}


__global__ void depth_pcloud_render_idx_kernel(float* pcloud, int* zbuf, int* idbuf, int batch_size, int n, int h, int w){
    int pid = blockIdx.x;
    int bid = blockIdx.y;
    float* p_data = pcloud + (bid*n*3 + pid*3);
    int x = max(0, min((int)(w*((*p_data)*0.5 + 0.5)), w-1));
    int y = max(0, min((int)(h*((*(p_data+1))*0.5 + 0.5)), h-1));
    for(int tx = max(0, x-1); tx < min(x+1, w); tx++){   
        for(int ty = max(0, y-1); ty < min(y+1, w); ty++){ 
            if((int)(*(p_data+2)*1e5) >= *(zbuf + (bid*h*w + w*y + x))){
                atomicExch(idbuf + (bid*h*w + w*y + x), bid*n + pid);
            }
        }
    }
}


void depth_pcloud_render_idx_cuda(float* pcloud, int* zbuf, int* idbuf, int batch_size, int n, int h, int w){
    // pcloud B N 3
    // zbuf B H W
    dim3 blocks_dim(n, batch_size);

    depth_pcloud_render_kernel<<<blocks_dim, 1>>>(
        pcloud, zbuf, batch_size, n, h, w
    );
    depth_pcloud_render_idx_kernel<<<blocks_dim, 1>>>(
        pcloud, zbuf, idbuf, batch_size, n, h, w
    );
}


__global__ void depth_pcloud_render_idx_backward_kernel(float* pcloud, int* idbuf, float* grad, float* grad_out, int batch_size, int n, int h, int w){
    int bid = blockIdx.x;
    int hid = blockIdx.y;
    int wid = blockIdx.z;
    int id_data = *(pcloud + (bid*h*w + hid*w + wid));
    if(id_data >= 0){
        atomicExch(grad_out + id_data, *(grad +  (bid*h*w + hid*w + wid)));
    }
}

void depth_pcloud_render_idx_backward_cuda(float* pcloud, int* idbuf, float* grad, float* grad_out, int batch_size, int n, int h, int w){
    dim3 blocks_dim(batch_size, h, w);

    depth_pcloud_render_idx_backward_kernel<<<blocks_dim, 1>>>(
        pcloud, idbuf, grad, grad_out, batch_size, n, h, w
    );
}