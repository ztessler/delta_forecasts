import json
import numpy as np
import pandas
import rasterio


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
    with open(str(source[1]), 'r') as f:
        upstream = json.load(f)
    with rasterio.open(str(source[2]), 'r') as data_rast:
        data = data_rast.read(1)
    if method in ['weightedsum', 'weightedmean']:
        with rasterio.open(str(source[3]), 'r') as weight_rast:
            weights = weight_rast.read(1)
    else:
        weights = np.ones_like(data)

    aggregated = {}
    for delta in upstream:
        ids = upstream[delta]['basin_ids']
        pixels = np.zeros_like(data, dtype=np.float)
        for basinid in ids:
            pixels = np.logical_or(pixels, basins==basinid)
        aggregated[delta] = aggregate(data[pixels], weights[pixels])

    pandas.Series(aggregated).to_pickle(str(target[0]))
    return 0


def convert_m_to_km(env, target, source):
    p = pandas.read_pickle(str(source[0]))
    (p / 1e3).to_pickle(str(target[0]))
    return 0