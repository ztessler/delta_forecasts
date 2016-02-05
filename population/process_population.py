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
        stats[dname] = {
                'pop_density': _stats['mean'],  # people/sq.km.
                'pop_count':  _stats['mean'] * delta.area.iloc[0] / 1e6,  # people
                }
    raster.close()
    with open(str(target[0]), 'w') as outfile:
        json.dump(stats, outfile)
    return 0


def clip_pop_to_delta(env, target, source):
    dname = env['delta']
    with open(str(source[0]), 'r') as f:
        delta = sgeom.shape(json.load(f))

    nodata = -9999

    with rasterio.open(str(source[1]), 'r') as src:
        kwargs = src.meta.copy()
        del kwargs['transform']

        mask = rasterize(delta, out_shape=src.shape, transform=src.affine, dtype=src.dtypes[0])
        window = rasterio.get_data_window(mask, 0)
        image = src.read(1, window=window)
        mask = mask[slice(*window[0]), slice(*window[1])]
        image[mask==0] = nodata

        kwargs.update({
            'height': window[0][1] - window[0][0],
            'width': window[1][1] - window[1][0],
            'affine': src.window_transform(window),
            'nodata': nodata})

        with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
            dst.write(image, 1)

        return 0
