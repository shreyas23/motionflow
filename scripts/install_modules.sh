#!/bin/bash
pip3 install spatial-correlation-sampler==0.2.1
cd ./models/correlation_package
python3 setup.py install
cd ../forwardwarp_package
python3 setup.py install
cd ../..