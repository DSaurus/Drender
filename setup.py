import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

include_dirs = torch.utils.cpp_extension.include_paths()
# print(include_dirs)
# include_dirs.append('/media/yanshi/windows/attention_kernel/src')
# include_dirs.append('/media/yanshi/windows/attention_kernel/pybind11-master/include')
# print(include_dirs)

setup(
    name="depth_render",
    version="0.0.1",
    description="rendering depth point cloud",
    # url="https://github.com/jbarker-nvidia/pytorch-correlation",
    author="Saurus",
    author_email="jia1saurus@gmail.com",
    ext_modules = [
        CUDAExtension(name='d_render', 
        include_dirs = include_dirs,
        sources=['src/d_render_kernel.cu', 'src/d_render.cpp'])
    ],
    cmdclass={
        'build_ext' : BuildExtension
    }
)
