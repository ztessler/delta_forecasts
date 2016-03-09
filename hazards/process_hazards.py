import os
import numpy as np
import pandas
import geopandas
import cartopy.crs as ccrs


def storm_surge_agg_points(env, target, source):
    deltas = geopandas.read_file(str(source[0])).set_index('Delta')
    surges = geopandas.read_file(str(source[1]))

    origcols = ['T_10', 'T_25', 'T_50', 'T_100', 'T_250']
    cols = map(lambda s: float(s[2:]), origcols)
    surges = surges[['geometry'] + origcols].rename_axis(
            {orig:new for orig, new in zip(['geometry']+origcols, ['geometry']+cols)},
            axis=1)

    centroids = deltas.centroid
    mean_surge = pandas.DataFrame(index=deltas.index, columns=cols)
    for dname in deltas.index:
        lon, lat = np.array(centroids.loc[dname])
        # reproject delta shape to Azimuthal Equidistant - distances are correct from center point, good for buffering
        aed = ccrs.AzimuthalEquidistant(central_longitude=lon,
                                        central_latitude=lat)
        delta = deltas.loc[[dname]].to_crs(aed.proj4_params)['geometry']
        # buffer around convex hull (very slow over all points)
        delta_buff = delta.convex_hull.buffer(25 * 1000) # 25km
        # project back to match surge data
        poly = delta_buff.to_crs(surges.crs).item()

        delta_surges = surges[surges.within(poly)][cols]
        mean_surge.loc[dname,:] = delta_surges[delta_surges>0].mean(axis=0)

    mean_surge.to_pickle(str(target[0]))
    return 0
