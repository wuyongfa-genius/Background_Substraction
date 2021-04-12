from setuptools import setup, find_packages

setup(name='Background Substraction',
      version='0.0.0',
      description='Some common algorithms to perform BGS',
      author='wuyongfa',
      author_email='',
      url='https://github.com/wuyongfa-genius/Background_Substraction',
      install_requires=['numpy', 'tifffile', 'imagecodecs', 'opencv-python'],
      packages=find_packages()
      )