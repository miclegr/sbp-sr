from setuptools import setup, find_packages

VERSION = "0.0"
NAME = 'sbpsr'
DESCRIPTION = "Implementation of Smooth Bilevel Programming for Sparse Regularization"
MAINTAINER = 'Michele Gregori'
MAINTAINER_EMAIL = 'michelegregorits@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/miclegr/sbpsr.git'
URL = 'https://github.com/miclegr/sbpsr'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.rst').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      install_requires=['numpy>=1.18', 'scipy>=1.4.1'],
      packages=find_packages(),
      )
