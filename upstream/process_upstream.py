import json
import numpy as np
import pandas
import rasterio


def make_rast_val(env, target, source):
    val = env['val']
    with rasterio.open(str(source[0]), 'r') as rast:
        meta = rast.meta.copy()
    assert val != meta['nodata']
    del meta['transform']
    with rasterio.open(str(target[0]), 'w', **meta) as dst:
        data = np.ones(dst.shape, dtype=type(val)) * val
        dst.write(data, 1)
    return 0

def set_upstream_val(env, target, source):
    basin_ids = pandas.read_pickle(str(source[0]))
    series = pandas.Series(env['val'], index=basin_ids.index)
    series.to_pickle(str(target[0]))
    return 0


def agg_over_basins(env, target, source):
    '''
    Aggregate raster values within all basins contributing to each delta.
    If method is weightedsum or weightedmean, then values are weighted using
    the weights raster

    For computing basin area, can either use a raster of all ones, area
    weights, and method='weightedsum', or use a raster of area and method='sum',
    sending in a dummy placeholder raster for weights.
    '''
    method = env['method']
    if method == 'sum':
        aggregate = lambda data, _: np.sum(data)
    elif method == 'weightedsum':
        aggregate = lambda data, weights: np.sum(data * weights)
    elif method == 'mean':
        aggregate = lambda data, _: np.mean(data)
    elif method == 'weightedmean':
        aggregate = lambda data, weights: np.sum(data * weights) / np.sum(weights)
    elif method == 'max':
        aggregate = lambda data, _: np.max(data)
    else:
        raise ValueError, 'method must be sum, weightedsum, mean, weightedmean, max'

    with rasterio.open(str(source[0]), 'r') as basins_rast:
        basins = basins_rast.read(1)
    basin_ids = pandas.read_pickle(str(source[1]))
    with rasterio.open(str(source[2]), 'r') as data_rast:
        data = data_rast.read(1, masked=True)
    if method in ['weightedsum', 'weightedmean']:
        with rasterio.open(str(source[3]), 'r') as weight_rast:
            weights = weight_rast.read(1)
    else:
        weights = np.ones_like(data)

    aggregated = []
    for delta, basinid in basin_ids.index:
        pixels = (basins==basinid)
        aggregated.append(aggregate(data[pixels], weights[pixels]))
    if np.ma.masked in aggregated:
        fill = env['fill']
        if fill == 'mean':
            fill = (data * weights).sum() / weights[~data.mask].sum()
    aggregated = [a if a is not np.ma.masked else fill for a in aggregated]
    pandas.Series(aggregated, index=basin_ids.index).to_pickle(str(target[0]))
    return 0


def convert_m_to_km(env, target, source):
    p = pandas.read_pickle(str(source[0]))
    (p / 1e3).to_pickle(str(target[0]))
    return 0


def clip_neg_to_zero(env, target, source):
    p = pandas.read_pickle(str(source[0]))
    p[p<0] = 0
    p.to_pickle(str(target[0]))
    return 0


def discharge_at_mouths(env, target, source):
    with rasterio.open(str(source[0]), 'r') as rast:
        globaldis = rast.read(1)
    mouths = pandas.read_pickle(str(source[1]))
    discharge = pandas.Series(index=mouths.index)

    for (delta, basin), mouth in mouths.iterrows():
        discharge.loc[delta, basin] = globaldis[mouth.y, mouth.x]

    discharge.to_pickle(str(target[0]))
    return 0
