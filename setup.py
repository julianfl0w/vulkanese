#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="vulkanese",
    version="1.0",
    description="An abstraction of Vulkan",
    author="Julian Loiacono",
    author_email="jcloiacon@gmail.com",
    url="https://github.com/julianfl0w/vulkanese",
    packages=find_packages(exclude=('examples')),
    package_data={
        # everything
        #"": ["*"]
        "": ["."]
    },
    include_package_data=True,
)
