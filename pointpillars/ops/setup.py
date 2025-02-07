from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointpillars',
    ext_modules=[
        CUDAExtension(
            name='voxel_op', 
            sources=['voxelization/voxelization.cpp',
                     'voxelization/voxelization_cpu.cpp',
                     'voxelization/voxelization_cuda.cu',
                    ],
            define_macros=[('WITH_CUDA', None)]    
        ),
        CUDAExtension(
            name='iou3d_op', 
            sources=['iou3d/iou3d.cpp',
                     'iou3d/iou3d_kernel.cu',
                    ],
            define_macros=[('WITH_CUDA', None)]    
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })