import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='my_project',
    version='1.0',
    author='enzo',
    ext_modules=[
        CUDAExtension(
            name='my_project_cuda_extension',
            sources=['cpp_example.cpp',
                     'cuda_example.cu'],
            include_dirs=[Path(this_dir) / "include"],
            extra_compile_args={'cxx': ["-O3", "-std=c++17"],
                                'nvcc':["-O3", "-std=c++17"]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)