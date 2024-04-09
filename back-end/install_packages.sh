#!/bin/sh

# DO NOT put the installation of pytorch in this file
# As indicated in the README file, users need to use:
# $conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# The reason is that users may have different cuda versions

# http related
pip install --upgrade requests

# OpenCV
pip install --upgrade opencv-python
pip install --upgrade opencv-contrib-python

# For plotting images
pip install --upgrade matplotlib

# For machine learning
pip install --upgrade scikit-learn

# For TensorBoard
pip install --upgrade tb-nightly
pip install --upgrade tensorflow
pip install --upgrade future
pip install --upgrade moviepy

# For data analysis
pip install --upgrade pandas

# For pytorch related
pip install --upgrade torchviz
pip install --upgrade torchsummary
