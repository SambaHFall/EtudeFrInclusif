import os

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

reqs = [line.strip() for line in open("requirements.txt").readlines() ]

setup(
    name='EtudeFrInclusif',
    packages=find_packages(),
    install_requires=reqs,

    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Samba Fall',
    url='https://github.com/SambaHFall/EtudeFrInclusif',
)
