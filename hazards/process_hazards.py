import os
import numpy as np
import pandas
import geopandas
import cartopy.crs as ccrs


def storm_surge_agg_points(env, target, source):
    deltas = geopandas.read_file(str(source[0])).set_index('Delta')
    surges = geopandas.read_file(str(source[1]))
    level = env['return_level']

    centroids = deltas.centroid
    mean_surge = pandas.Series(index=deltas.index)
    for dname in deltas.index:
        # delta location
        lon, lat = np.array(centroids.loc[dname])
        # d_minlon, d_minlat, d_maxlon, d_maxlat = bounds.loc[dname]

        # reproject delta shape to Azimuthal Equidistant - distances are correct from center point
        aed = ccrs.AzimuthalEquidistant(central_longitude=lon,
                                        central_latitude=lat)
        delta = deltas.loc[[dname]].to_crs(aed.proj4_params)['geometry']
        # buffer in aed and project back to match surge data
        delta_buff = delta.convex_hull.buffer(25 * 1000) # 25km
        poly = delta_buff.to_crs(surges.crs).item()

        delta_surges = surges[surges.within(poly)][level]
        mean_surge[dname] = delta_surges[delta_surges>0].mean()

    mean_surge.fillna(0).to_pickle(str(target[0]))
    return 0
