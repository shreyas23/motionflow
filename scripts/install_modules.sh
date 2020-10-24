#!/bin/bash
pip3 install spatial-correlation-sampler -vvv
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension/
python setup.py install
cd ../
cd ./models/correlation_package
python3 setup.py install
cd ../forwardwarp_package
python3 setup.py install
cd ../..