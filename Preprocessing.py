#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : Preprocessing.py
# description     : Import and preprocess data
# author          : Nicolas Coucke
# date            : 2022-07-05
# version         : 1
# usage           : python helloworld.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.13
# ==============================================================================

print('start test again')

import io
from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mne
import requests

from hypyp import (
    prep,
)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import stats
from hypyp import viz

print('everything imported')



