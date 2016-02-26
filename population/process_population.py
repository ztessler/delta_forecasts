import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import geopandas
from affine import Affine
import rasterio
from rasterio.features import rasterize
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
    pandas.DataFrame.from_dict(stats, orient='index').to_pickle(str(target[0]))
    return 0


def clip_pop_to_delta(env, target, source):
    delta = geopandas.read_file(str(source[0]))

    with rasterio.open(str(source[1]), 'r') as src:
        kwargs = src.meta.copy()
        del kwargs['transform']

        mask = rasterize(delta.loc[0, 'geometry'], default_value=1, fill=0, out_shape=src.shape, transform=src.affine, dtype=src.dtypes[0])
        window = rasterio.get_data_window(mask, 0)
        image = src.read(1, window=window)
        mask = mask[slice(*window[0]), slice(*window[1])]
        image[mask==0] = src.nodata

        kwargs.update({
            'height': window[0][1] - window[0][0],
            'width': window[1][1] - window[1][0],
            'affine': src.window_transform(window)})

        with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
            dst.write(image, 1)

        return 0

def pop_elevation_bins(env, target, source):
    delta = geopandas.read_file(str(source[0]))
    lon, lat = np.array(delta.centroid.squeeze())
    minlon, minlat, maxlon, maxlat = np.array(delta.bounds.squeeze())

    laea = ccrs.LambertAzimuthalEqualArea(central_longitude=lon,
                                          central_latitude=lat)
    area_sqkm = delta.to_crs(laea.proj4_params)['geometry'].area.squeeze() / 1e6

    with rasterio.open(str(source[1]), 'r') as srtm_fd:
        srtm_raw = srtm_fd.read(1)
        srtm_raw_crs = srtm_fd.crs
        srtm_raw_affine = srtm_fd.affine
        srtm_raw_width = srtm_fd.width
        srtm_raw_height = srtm_fd.height
        srtm_raw_nodata = srtm_fd.nodata

    with rasterio.open(str(source[2]), 'r') as pop_fd:
        kwargs = pop_fd.meta.copy()
        pop_raw = pop_fd.read(1)
        pop_raw_crs = pop_fd.crs
        pop_raw_affine = pop_fd.affine
        pop_raw_bounds = pop_fd.bounds
        pop_raw_width = pop_fd.width
        pop_raw_height = pop_fd.height
        pop_raw_nodata = pop_fd.nodata

    # estimate reprojection params and pixel sizes based on population grid
    dst_crs = laea.proj4_params
    dst_affine, dst_width, dst_height = calculate_default_transform(
            pop_raw_crs, dst_crs, pop_raw_width, pop_raw_height,
            *pop_raw_bounds)

    pop = np.ones((dst_height, dst_width), dtype=rasterio.float64)
    srtm = np.ones((dst_height, dst_width), dtype=rasterio.float64)

    reproject(pop_raw, pop, pop_raw_affine, pop_raw_crs, pop_raw_nodata,
            dst_affine, dst_crs, pop_raw_nodata, RESAMPLING.bilinear)
    reproject(srtm_raw, srtm, srtm_raw_affine, srtm_raw_crs, srtm_raw_nodata,
            dst_affine, dst_crs, srtm_raw_nodata, RESAMPLING.bilinear)

    good = np.logical_and(pop != pop_raw_nodata, srtm != srtm_raw_nodata)
    pops = {}
    elevs = range(35+1)
    for elev in elevs:
        under = np.logical_and(good, srtm <= elev)
        over = np.logical_and(good, srtm > elev)
        frac_under = under.sum() / float(good.sum())
        pops[elev] = pop[under].mean() * frac_under * area_sqkm
    pandas.Series(pops, name='Population').to_pickle(str(target[0]))
    return 0

def plot_hypsometric(env, target, source):
    pops = pandas.read_pickle(str(source[0]))
    plt.style.use('ggplot')
    f, a = plt.subplots(1, 1)
    pops.plot(ax=a, title=env['delta'])
    a.set_xlabel('Elevation, m')
    a.set_ylabel('Population at or below elevation')
    f.savefig(str(target[0]))
    return 0
