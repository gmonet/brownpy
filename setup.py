from setuptools import setup, find_packages

# https://blog.godatadriven.com/setup-py

setup(
    name='brownpy',
    version='0.2.1b',
    description='Brownian motion through nanochannel',
    author='Geoffrey Monet',
    packages=find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy>=1.16.5',
        'scipy>=1.3.1',
        'matplotlib>=3',
        'numba>=0.45',
        'tqdm>=4',
        'h5py>=2',
    ],
    include_package_data=True,
)
