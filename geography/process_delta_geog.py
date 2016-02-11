import numpy as np
import json
import geopandas
import shapely.geometry as sgeom
from rasterstats import zonal_stats
from collections import OrderedDict

def group_delta_shps(env, target, source):
    shpfile = str(source[0])

    deltas = geopandas.GeoDataFrame.from_file(shpfile)
    crs = deltas.crs

    deltas = deltas.groupby('Delta')\
                   .aggregate({
                       'DeltaID': lambda s: s.iloc[0],
                       'geometry': lambda s: sgeom.MultiPolygon(list(s)),
                        })
    deltas = geopandas.GeoDataFrame(deltas)
    deltas['Delta'] = deltas.index #index lost on saving to file
    deltas.crs = crs

    deltas.to_file(str(target[0]), 'ESRI Shapefile')
    return 0

def delta_geojson(env, target, source):
    dname = env['delta']
    deltas = geopandas.GeoDataFrame.from_file(str(source[0])).set_index('Delta')
    delta = deltas.loc[[dname]]
    delta.reset_index().to_file(str(target[0]), 'GeoJSON')

    return 0

def contributing_basins(env, target, source):
    def ma_unique_values(ma):
        return {int(b) for b in ma[np.logical_not(ma.mask)]}

    stats = zonal_stats(
            str(source[0]), str(source[1]),
            geojson_out=True,
            add_stats={'basins': ma_unique_values})

    data = OrderedDict()
    for s in stats:
        deltainfo = OrderedDict()
        deltainfo['DeltaID'] = int(s['properties']['DeltaID'])
        deltainfo['basin_ids'] = sorted(s['properties']['basins'])
        data[s['properties']['Delta']] = deltainfo

    with open(str(target[0]), 'w') as outfile:
        json.dump(data, outfile)
    return 0


