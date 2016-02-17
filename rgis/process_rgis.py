import numpy as np
from netCDF4 import Dataset
import rasterio
from affine import Affine

def georef_nc(env, target, source):
    nc = Dataset(str(source[0]))
    var = nc.variables[nc.subject]
    data = var[:].squeeze()
    nodata = var.missing_value
    lat_bnds = nc.variables['latitude_bnds'][:]
    lon_bnds = nc.variables['longitude_bnds'][:]

    yoff, xoff = data.shape
    sx = np.diff(lon_bnds).mean()
    sy = np.diff(lat_bnds).mean()

    affine = Affine.scale(sx, -sy) * Affine.translation(-xoff/2., -yoff/2.)

    with rasterio.open(str(target[0]), 'w',
            driver='GTiff', width=xoff, height=yoff,
            crs={'init':'epsg:4326'}, transform=affine,
            count=1, nodata=nodata, dtype=data.dtype) as dst:
        dst.write(np.flipud(data), 1)

    nc.close()
    return 0
