import os
from setuptools import setup, find_packages

from distutils.command.install import INSTALL_SCHEMES
for scheme in INSTALL_SCHEME.values() :
	scheme['data'] = scheme['purelib']

with open('README.md') as f:
    long_description = f.read()

reqs = [line.strip() for line in open("requirements.txt").readlines() ]

setup(
    name='EtudeFrInclusif',
    packages=find_packages(),
    install_requires=reqs,

	include_package_data = True
	
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Samba Fall',
    url='https://github.com/SambaHFall/EtudeFrInclusif',
)
