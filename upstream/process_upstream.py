import json
import numpy as np
import pandas
import rasterio

def agg_over_basins(env, target, source):
    with rasterio.open(str(source[0]), 'r') as basins_rast:
        basins = basins_rast.read(1)
    with rasterio.open(str(source[1]), 'r') as rast:
        data = rast.read(1)
    with open(str(source[2]), 'r') as f:
        upstream = json.load(f)
    method = env['method']

    if method == 'sum':
        aggregate = np.sum
    elif method == 'mean':
        aggregate = np.mean
    else:
        raise ValueError, 'method must be sum or mean'

    aggregated = {}
    for delta in upstream:
        ids = upstream[delta]['basin_ids']
        pixels = np.zeros_like(data, dtype=np.float)
        for basinid in ids:
            pixels = np.logical_or(pixels, basins==basinid)
        aggregated[delta] = aggregate(data[pixels])

    pandas.Series(aggregated).to_pickle(str(target[0]))
    return 0



