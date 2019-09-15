# -*- coding: utf-8 -*-
"""
Setup script.
Created on Sun Sep 15 20:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/openpose-pytorch

"""


from setuptools import setup, find_packages


setup(name='openpose-pytorch',
      version='0.1.0',
      description='PyTorch implementation of OpenPose',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Prasun Roy',
      author_email='prasunroy.pr@gmail.com',
      url='https://github.com/prasunroy/openpose-pytorch',
      license='MIT',
      install_requires=[
              'numpy',
              'opencv-contrib-python',
              'scipy',
              'tqdm'
      ],
      classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.7',
              'Topic :: Scientific/Engineering',
              'Topic :: Software Development :: Libraries',
              'Topic :: Software Development :: Libraries :: Python Modules',
              'Topic :: Utilities'
      ],
      keywords=[
              'openpose',
              'pytorch',
              'pose-estimation',
              'keypoint-estimation',
              'computer-vision',
              'machine-learning',
              'openpose-wrapper',
              'pytorch-implementation'
      ],
      packages=find_packages())
