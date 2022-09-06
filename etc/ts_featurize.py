"""
Module to interface with  TSFEL to extract features from time series and add it to features datafram
"""
import os, sys
import pandas as pd
import numpy as np
import tsfel

x = np.random.rand((1000, 3))
X = pd.DataFrame(X)
