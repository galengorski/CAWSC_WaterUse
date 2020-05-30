from setuptools import setup, find_packages
import sys
import platform

if sys.version_info < (3, 0):
    raise EnvironmentError("Python 3 is needed for this package")

requirements = ["pandas",
                "numpy",
                "beautifulsoup4",
                "pyshp",
                "pycrs",
                "geojson",
                "shapely",
                "requests",
                "scipy"]

if platform.system().lower() != "windows":
    requirements.append("ray")


setup_requirements = []
test_requirements = []
description = """Python package to query census population data and provide
estimates of population from census data from arbitrary polygons"""

setup(
    author="Joshua Larsen",
    author_email='jlarsen@usgs.gov',
    classifiers=[
        'Development Status :: 2 Beta',
        'Intended Audience :: Scientists',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: GIS'
    ],
    description=description,
    install_requires=requirements,
    license="MIT License",
    long_description=description,
    include_package_data=True,
    keywords="census",
    name='censusdc',
    packages=find_packages(include=['censusdc',
                                    'censusdc.datacollector',
                                    'censusdc.utils']),
    setup_requires=setup_requirements,
    url='',
    version='0.2.0',
    zip_safe=False
)
