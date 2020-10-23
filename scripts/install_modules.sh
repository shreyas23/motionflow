#!/bin/bash
pip install spatial-correlation-sampler
pip3 install spatial-correlation-sampler
cd ./models/correlation_package
python3 setup.py install
cd ../forwardwarp_package
python3 setup.py install
cd ../..