from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name='my_project',
    version='1.0',
    author='enzo',
    ext_modules=[
        CppExtension(
            name='my_project_cpp_extension',
            sources=['example.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)