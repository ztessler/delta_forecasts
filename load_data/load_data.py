import os
import fiona
import geopandas
import shapely.geometry as sgeom

def group_delta_shps(env, target, source):
    shpfile = str(source[0])

    # with fiona.open(shpfile) as shp:
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
    delta.to_file(str(target[0]), 'GeoJSON')

    return 0



