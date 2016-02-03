import os
import json
import numpy as np
import geopandas
from affine import Affine
import rasterio
from rasterio.warp import calculate_default_transform, reproject, RESAMPLING
import cartopy.crs as ccrs
from rasterstats import zonal_stats

def delta_population_stats(env, target, source):
    # reprojects delta shape to LamberAzimuthalEqualArea projection
    # centered on delta.  Extracts data from (global) raster, warps to same
    # projection and calculates zonal stats within delta
    year = env['year']
    deltas = geopandas.GeoDataFrame.from_file(str(source[0]))
    deltas = deltas.set_index('Delta')
    raster = rasterio.open(str(source[1]))
    nodata = raster.nodata
    t = raster.affine
    
    centroids = deltas.centroid
    bounds = deltas.bounds
    stats = {}
    for dname in deltas.index:
        lon, lat = np.array(centroids.loc[dname])
        minlon, minlat, maxlon, maxlat = bounds.loc[dname]
        laea = ccrs.LambertAzimuthalEqualArea(central_longitude=lon,
                                              central_latitude=lat)
        delta = deltas.loc[[dname]].to_crs(laea.proj4_params)['geometry']

        maxy, minx = raster.index(minlon, minlat)
        miny, maxx = raster.index(maxlon, maxlat)
        window = ((miny, maxy+1), (minx, maxx+1))
        src = raster.read(1, window=window)
        src_height, src_width = src.shape
        src_bounds = raster.window_bounds(window)
        src_affine = Affine(t.a, t.b, t.c+minx*t.a, t.d, t.e, t.f+miny*t.e)

        dst_affine, dst_width, dst_height = calculate_default_transform(
                raster.crs, laea.proj4_params, src_width, src_height,
                *src_bounds)
        dst = np.ones((dst_height, dst_width)) * nodata
        reproject(src, dst, src_affine, raster.crs, nodata, dst_affine,
                laea.proj4_params, nodata, RESAMPLING.bilinear)

        _stats = zonal_stats(delta, dst, affine=dst_affine, nodata=nodata)
        _stats = _stats[0]   # delta only has a single Multipolygon feature
        stats[dname] = {
                'pop_density': _stats['mean'],  # people/sq.km.
                'pop_count':  _stats['mean'] * delta.area.iloc[0] / 1e6,  # people
                }
    raster.close()
    with open(str(target[0]), 'w') as outfile:
        json.dump(stats, outfile)
    return 0



