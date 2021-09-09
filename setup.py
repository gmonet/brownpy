from setuptools import setup, find_packages

# https://blog.godatadriven.com/setup-py

setup(
    name='brownpy',
    version='0.0.2b',
    description='Brownian motion through nanochannel',
    author='Geoffrey Monet',
    packages=find_packages(include=['channel-bm', 'channel-bm.*']),
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy>=1.16.5',
        'scipy>=1.3.1',
        'matplotlib>=3.1.1',
        'numba>=0.45.1',
        'tqdm>=4.36.1',
        'netCDF4>=3',
    ],
    include_package_data=True,
)
