import os
import fiona
import geopandas
import shapely.geometry as sgeom

def import_deltas(dependencies, targets):
    # shpfile = os.path.join(os.environ['HOME'], 'data', 'deltas', 'global_map_shp/global_map.shp') 
    shpfile = dependencies.pop()

    # with fiona.open(shpfile) as shp:
    deltas = geopandas.GeoDataFrame.from_file(shpfile)
    crs = deltas.crs

    deltas = deltas.groupby('Delta')\
                   .aggregate({
                       'DeltaID': lambda s: s.iloc[0],
                       'geometry': lambda s: sgeom.MultiPolygon(list(s)),
                        })
    deltas = geopandas.GeoDataFrame(deltas)
    deltas.crs = crs

    if not os.path.exists(os.path.dirname(targets[0])):
        os.makedirs(os.path.dirname(targets[0]))
    deltas.to_file(targets[0])
