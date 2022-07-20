#include <THC/THC.h>
#include <torch/extension.h>
// #include <pybind11/pybind11.h>
#include "d_render.h"

extern THCState *state;

int depth_pcloud_render(torch::Tensor xyz_pcloud, torch::Tensor zbuf){
    // xyz_pcloud B N 3 zbuf B H W
    int batch_size = xyz_pcloud.size(0);
    int n = xyz_pcloud.size(1);
    int h = zbuf.size(1);
    int w = zbuf.size(2);
    depth_pcloud_render_cuda(xyz_pcloud.data_ptr<float>(), zbuf.data_ptr<int>(), batch_size, n, h, w);
    return 0;
}


int depth_pcloud_render_idx(torch::Tensor xyz_pcloud, torch::Tensor zbuf, torch::Tensor idbuf){
    // xyz_pcloud B N 3 zbuf B H W
    int batch_size = xyz_pcloud.size(0);
    int n = xyz_pcloud.size(1);
    int h = zbuf.size(1);
    int w = zbuf.size(2);
    depth_pcloud_render_idx_cuda(xyz_pcloud.data_ptr<float>(), zbuf.data_ptr<int>(), idbuf.data_ptr<int>(), batch_size, n, h, w);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_pcloud_render", &depth_pcloud_render, "render");
  m.def("depth_pcloud_render_idx", &depth_pcloud_render_idx, "render");
}