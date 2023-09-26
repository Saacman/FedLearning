#!/bin/bash

python3.8 resnet_quant_dyn.py 1 1 8 4 8
python3.8 resnet_quant_dyn.py 2 2 8 6 8
python3.8 resnet_quant_dyn.py 2 2 10 6 10
python3.8 resnet_quant_dyn.py 2 2 12 6 12
python3.8 resnet_quant_dyn.py 2 2 8 6 8
python3.8 resnet_quant_dyn.py 2 2 10 6 10
python3.8 resnet_quant_dyn.py 2 2 12 6 12

# Add more lines as needed for different arguments
