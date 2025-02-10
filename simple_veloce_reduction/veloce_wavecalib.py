from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
# from csaps import csaps

from . import veloce_config, veloce_reduction_tools

def load_wave_calibration_for_interpolation():
    raise NotImplementedError

def interpolate_wave(orders, hdr):
    raise NotImplementedError

def calibrate_simLC():
    raise NotImplementedError

def calibrate_simTh():
    raise NotImplementedError