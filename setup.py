import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='wifi',
    version='0.0.1',
    url='https://github.com/gasparka/wifi',
    author='Gaspar Karm',
    author_email='gkarm@live.com',
    description='Wifi PHY implementation in Python',
    packages=find_packages(),
    install_requires=['pytest',
                      'pytest-cov',
                      'numpy',
                      'scipy',
                      'dataclasses',
                      'numba',
                      'hypothesis',
                      'loguru'],
)
