#!/bin/bash
# pip install spatial-correlation-sampler
# pip3 install spatial-correlation-sampler
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
python3 ./Pytorch-Correlation-extension/setup.py install
cd ./models/correlation_package
python3 setup.py install
cd ../forwardwarp_package
python3 setup.py install
cd ../..