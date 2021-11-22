from setuptools import setup, find_packages
import sys
import os

if sys.version_info >= (3, 0):
    requirements = ["pandas",
                    "numpy",
                    "pyshp",
                    "geojson",
                    "plotly"]
else:
    raise EnvironmentError("pyGSFLOW is only supported with python 3")

setup_requirements = []

test_requirements = []
long_description = ""

setup(
    author="Joshua Larsen",
    author_email='jlarsen@usgs.gov',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Hydrology',
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6",
    description="html_report is a package for creating interactive"
                " reports and plots that can be shared across systems",
    install_requires=requirements,
    license="MIT license",
    long_description=long_description,
    include_package_data=True,
    keywords='html_report',
    name='html_report',
    packages=find_packages(include=['html_report',
                                    'html_report.utils']),
    setup_requires=setup_requirements,
    test_suite='autotest',
    tests_require=test_requirements,
    url='https://github.com/jlarsen-usgs/html_reports',
    version='0.1',
    zip_safe=False,
)