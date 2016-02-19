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

def regrid_to_06min(env, target, source):
    dst_dx = 0.1
    dst_dy = -0.1
    dst_width = 3600
    dst_height = 1800
    dst_affine = (Affine.scale(dst_dx, dst_dy) *
                  Affine.translation(-dst_width/2., -dst_height/2.))
    data06 = np.zeros((dst_height, dst_width))

    with rasterio.open(str(source[0]), 'r') as src:
        reproject(src.read(1),
                  data06,
                  src_transform=src.affine,
                  src_crs=src.crs,
                  src_nodata=src.nodata,
                  dst_transform=dst_affine,
                  dst_crs=src.crs,
                  dst_nodata=src.nodata,
                  resampling=RESAMPLING.bilinear)
        kwargs = src.meta

    kwargs.update({
	    'transform': dst_affine,
	    'affine': dst_affine,
	    'width': dst_width,
	    'height': dst_height,
	})

    with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
        dst.write(data06, 1)
    return 0


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

    affine = Affine.translation(lon_bnds.min(), lat_bnds.max()) * Affine.scale(sx, -sy)

    with rasterio.open(str(target[0]), 'w',
            driver='GTiff', width=xoff, height=yoff,
            crs={'init':'epsg:4326'}, transform=affine,
            count=1, nodata=nodata, dtype=data.dtype) as dst:
        dst.write(np.flipud(data), 1)

    nc.close()
    return 0
