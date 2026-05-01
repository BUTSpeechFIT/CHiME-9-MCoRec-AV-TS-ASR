#!/bin/bash

find ./experiments -type f -name "*.yaml" -exec python script/run_evaluation.py {} \;
