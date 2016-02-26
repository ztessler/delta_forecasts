import numpy as np
import rasterio
from rasterio.warp import RESAMPLING, reproject
from affine import Affine
from netCDF4 import Dataset
import cartopy.crs as ccrs
import shapely.geometry as sgeom


def regrid_to_06min(env, target, source):
    dst_dx = 0.1
    dst_dy = -0.1
    dst_width = 3600
    dst_height = 1800
    dst_affine = (Affine.scale(dst_dx, dst_dy) *
                  Affine.translation(-dst_width/2., -dst_height/2.))
    data06 = np.zeros((dst_height, dst_width))

    with rasterio.open(str(source[0]), 'r') as src:
        kwargs = src.meta
        if src.affine == dst_affine and src.shape == (dst_height, dst_width):
            data06 = src.read(1)
        else:
            reproject(src.read(1),
                      data06,
                      src_transform=src.affine,
                      src_crs=src.crs,
                      src_nodata=src.nodata,
                      dst_transform=dst_affine,
                      dst_crs=src.crs,
                      dst_nodata=src.nodata,
                      resampling=RESAMPLING.bilinear)

    kwargs.update({
	    'transform': dst_affine,
	    'affine': dst_affine,
	    'width': dst_width,
	    'height': dst_height,
	})

    with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
        dst.write(data06, 1)
    return 0


def regrid_to_raster(env, target, source):
    resampling = getattr(RESAMPLING, env['method'])

    with rasterio.open(str(source[0]), 'r') as rast:
        meta = rast.meta
        dst_shape = rast.shape
    newdata = np.zeros(dst_shape)
    del meta['transform']

    with rasterio.open(str(source[1]), 'r') as src:
        reproject(src.read(1),
                  newdata,
                  src_transform=src.affine,
                  src_crs=src.crs,
                  src_nodata=src.nodata,
                  dst_transform=meta['affine'],
                  dst_crs=meta['crs'],
                  dst_nodata=src.nodata,
                  resampling=resampling)
    meta.update({
        'nodata': src.nodata,
        })

    with rasterio.open(str(target[0]), 'w', **meta) as dst:
        dst.write(newdata, 1)
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


def raster_pixel_areas(env, target, source):
    with rasterio.open(str(source[0]), 'r') as rast:
        shape = rast.shape
        rastcrs = rast.crs
        affine = rast.affine
        kwargs = rast.meta
    del kwargs['transform']
    areas = np.zeros(shape, dtype=np.float)

    if rastcrs['init'] == 'epsg:4326':
        geod = ccrs.PlateCarree().as_geodetic()
    else:
        raise NotImplementedError, 'Only works with lat/lon grid, assumes pixels rows have constant area'

    lon0, lat0 = affine * (0,0)
    lon1, lat1 = affine * shape[::-1]
    dlon_2 = affine.a / 2.
    dlat = affine.e
    dlat_2 = dlat / 2.
    lats = np.arange(lat0, lat1, dlat)
    for j, lat in enumerate(lats):
        lat_next = lat + dlat
        laea = ccrs.LambertAzimuthalEqualArea(central_longitude=0,
                                              central_latitude=lat+dlat_2)
        poly = sgeom.Polygon(
                laea.transform_points(geod,
                                      np.array([-dlon_2, -dlon_2, dlon_2, dlon_2]),
                                      np.array([lat, lat_next, lat_next, lat])))
        areas[j,:] = poly.area / 1e6

    with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
        dst.write(areas, 1)
    return 0
