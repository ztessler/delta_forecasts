import numpy as np
import pandas
import geopandas
import rasterio
from rasterio.warp import RESAMPLING, reproject, calculate_default_transform
from affine import Affine
from netCDF4 import Dataset
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from rasterstats import zonal_stats


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


def delta_zonal_stats(env, target, source):
    # reprojects delta shape to LamberAzimuthalEqualArea projection
    # centered on delta.  Extracts data from (global) raster, warps to same
    # projection and calculates zonal stats within delta
    deltas = geopandas.GeoDataFrame.from_file(str(source[0])).set_index('Delta')
    raster = rasterio.open(str(source[1]))
    nodata = raster.nodata
    minlon, minlat, maxlon, maxlat = raster.bounds

    centroids = deltas.centroid
    bounds = deltas.bounds
    stats = {}
    for dname in deltas.index:
        # delta location
        lon, lat = np.array(centroids.loc[dname])
        d_minlon, d_minlat, d_maxlon, d_maxlat = bounds.loc[dname]

        # reproject delta shape
        laea = ccrs.LambertAzimuthalEqualArea(central_longitude=lon,
                                              central_latitude=lat)
        delta = deltas.loc[[dname]].to_crs(laea.proj4_params)['geometry']

        # extract minimum raster around delta
        d_maxy, d_minx = raster.index(d_minlon, d_minlat)
        d_miny, d_maxx = raster.index(d_maxlon, d_maxlat)
        window = ((d_miny, d_maxy+1), (d_minx, d_maxx+1))
        src = raster.read(1, window=window)

        # get size, bounds, affine of raster subset
        src_height, src_width = src.shape
        src_bounds = raster.window_bounds(window)
        src_affine = raster.window_transform(window)

        # reproject raster
        dst_affine, dst_width, dst_height = calculate_default_transform(
                raster.crs, laea.proj4_params, src_width, src_height,
                *src_bounds)
        dst = np.ones((dst_height, dst_width)) * nodata
        reproject(src, dst, src_affine, raster.crs, nodata, dst_affine,
                laea.proj4_params, nodata, RESAMPLING.bilinear)

        # calculate zonal stats
        _stats = zonal_stats(delta, dst, affine=dst_affine, nodata=nodata)
        _stats = _stats[0]   # delta only has a single Multipolygon feature
        stats[dname] = _stats
    raster.close()
    pandas.DataFrame.from_dict(stats, orient='index').to_pickle(str(target[0]))
    return 0


def multiply_rast(env, target, source):
    factor = env['factor']
    with rasterio.open(str(source[0]), 'r') as src:
        kwargs = src.meta.copy()
        data = src.read(1)
    del kwargs['transform']

    with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
        dst.write(factor * data, 1)
    return 0
