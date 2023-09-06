#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="vfa",
    version=0.2,
    author="Harry10459",
    url="https://github.com/Harry10459/vfa_vqvae",
    description="Codebase for few-shot object detection",
    python_requires=">=3.6",
    packages=find_packages(exclude=('configs', 'data', 'work_dirs')),
    install_requires=[
        'clip@git+ssh://git@github.com/openai/CLIP.git'
    ],
)
