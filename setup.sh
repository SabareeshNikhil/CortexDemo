#! /bin/sh

virtualenv cortex_venv
source cortex_venv/bin/activate
pip install opencv_python==3.4.5.20
pip install matplotlib==2.2.4
pip install numpy==1.16.1

python cortex_demo.py
