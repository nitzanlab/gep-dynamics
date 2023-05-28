from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gepdynamics',
    packages=find_packages(),
    author='Yotam Avidar-Constantini',
    install_requires=requirements,
    extras_require={
        "gpu": ["torch>=1.0", "torchnmf>=0.3"],
    }
)