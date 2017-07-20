#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.10',
    'pint>=0.7.0',
    'xarray>=0.8.0',
    'six',
]

test_requirements = [
    'pytest>=2.9.2',
    'mock>=2.0.0',
]

setup(
    name='sympl',
    version='0.2.1',
    description='Sympl is a Toolkit for building Earth system models in Python.',
    long_description=readme + '\n\n' + history,
    author="Jeremy McGibbon",
    author_email='mcgibbon@uw.edu',
    url='https://github.com/mcgibbon/sympl',
    packages=['sympl'],
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=True,
    keywords='sympl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
