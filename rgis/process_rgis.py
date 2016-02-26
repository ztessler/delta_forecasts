import shutil
import numpy as np
import pandas
from netCDF4 import Dataset
import rasterio
from rasterio.warp import reproject, RESAMPLING
from affine import Affine


def tsv_to_pandas(env, target, source):
    p = pandas.read_csv(str(source[0]), sep='\t')
    p = p.set_index('ID')
    p.to_pickle(str(target[0]))
    return 0


def clip_nc_neg(env, target, source):
    shutil.copy(str(source[0]), str(target[0]))
    nc = Dataset(str(target[0]), 'a')
    data = nc.variables[env['varname']][:]
    mask = data.mask
    data[np.logical_and(data<0, ~data.mask)] = 0
    nc.variables[env['varname']][:] = data
    nc.close()
    return 0
