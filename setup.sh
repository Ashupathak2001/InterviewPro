#!/bin/bash

# Update pip and install base requirements
pip install --upgrade pip wheel setuptools
pip install --no-cache-dir numpy==1.23.5
pip install --no-cache-dir -r requirements.txt
python -m spacy download en_core_web_sm