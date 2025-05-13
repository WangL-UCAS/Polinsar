import numpy as np
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt
from osgeo import gdal
import mlutilooking
def remove_ground(sint_file, Omage,baseline):

    sint = gdal.Open(sint_file,gdal.GA_ReadOnly)
    sint = sint.GetRasterBand(1).ReadAsArray()
    sint = mlutilooking.multilook_float_numba(sint,2,4)
    if baseline > 0:
        sint = np.exp(1j * sint)
    else:
        sint = np.exp(-1j * sint)
    Omage12_flat = Omage * sint[:,:,np.newaxis,np.newaxis]

    return Omage12_flat




