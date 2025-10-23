#import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import re


#print('\nGPU Available: {}\n'.format(tf.config.list_physical_devices('GPU')))

testset=Dataset("../DATA/coupled_A13/obs_p8_010/nocorr/assim.nc")

print(testset)

quit()
