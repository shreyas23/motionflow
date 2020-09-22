#!/bin/bash
cd ./models/correlation_package
python3 setup.py install
cd ../forwardwarp_package
python3 setup.py install
cd ../..
