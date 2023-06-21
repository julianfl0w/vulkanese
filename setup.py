#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="vulkanese",
    version="1.0",
    description="An abstraction of Vulkan",
    author="Julian Loiacono",
    author_email="jcloiacon@gmail.com",
    url="https://github.com/julianfl0w/vulkanese",
    packages=find_packages(exclude=("examples")),
    install_requires=[
        'numpy',
        'opencv-python',
        'pysdl2',
        'pysdl2-dll',
        'open3d',
        'librosa',
        'screeninfo',
        'vulkan @ git+https://github.com/julianfl0w/vulkan',
        'sinode @ git+https://github.com/julianfl0w/sinode'
    ],

    package_data={
        # everything
        # "": ["*"]
        "": ["."]
    },
    include_package_data=True,
)
