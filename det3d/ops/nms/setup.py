from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='det3d.ops.nms.nms',
    ext_modules=[
        CUDAExtension('nms', [
            'nms.cc',
            'nms_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
