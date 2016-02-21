import json
import numpy as np
import pandas
import rasterio


def compute_I(env, target, source):
    Ag = pandas.read_pickle(str(source[0]))
    I = (1 + 0.09*Ag)
    I.to_pickle(str(target[0]))
    return 0

# env.Command(
        # source=['#data/Global/reservoir.tif',
                # '#data/Global/discharge.tif'],
        # target='#data/Global/trapping_eff_points.tif',
def res_trapping(env, target, source):
    utilization = 0.67
    with rasterio.open(str(source[0]), 'r') as resrast,\
             rasterio.open(str(source[1]), 'r') as disrast,\
             rasterio.open(str(source[2]), 'r') as basinrast:
        kwargs = resrast.meta.copy()
        resvol = resrast.read(1, masked=True) * utilization * (1000**3) # convert km**3 to m**3
        dis = disrast.read(1, masked=True)  # m**3 / s
        basins = basinrast.read(1, masked=True)
    with open(str(source[3]), 'r') as f:
        delta_basins = json.load(f)
    residence_time = resvol / dis / (60 * 60 * 24 * 365) # years
    trapping_eff = 1 - 0.05/np.sqrt(residence_time)

    Te = []
    keys = []
    for delta, basin_ids in delta_basins.iteritems():
        for basin_id in basin_ids:
            pix = np.logical_and(basins == basin_id, ~trapping_eff.mask)
            Te.append((trapping_eff[pix] * dis[pix]).sum() / dis[pix].sum())
            keys.append((delta, basin_id))
    Te = [te if (te is not np.ma.masked and te >= 0) else 0 for te in Te]

    index = pandas.MultiIndex.from_tuples(keys)
    Te = pandas.Series(Te, index=index)
    Te.to_pickle(str(target[0]))
    return 0
