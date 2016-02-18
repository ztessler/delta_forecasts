import numpy as np
import json
import geopandas
import rasterio
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from rasterstats import zonal_stats
from collections import OrderedDict


def clean_delta_name(delta):
    return delta.replace(' ','_').replace('-','_')


def group_delta_shps(env, target, source):
    deltas = geopandas.GeoDataFrame.from_file(str(source[0]))
    crs = deltas.crs

    deltas = deltas.groupby('Delta')\
                   .aggregate({
                       'DeltaID': lambda s: s.iloc[0],
                       'geometry': lambda s: sgeom.MultiPolygon(list(s)),
                        })
    deltas = geopandas.GeoDataFrame(deltas)
    deltas = deltas.rename(index={d: clean_delta_name(d) for d in deltas.index})
    deltas['Delta'] = deltas.index #index lost on saving to file
    deltas.crs = crs

    deltas.to_file(str(target[0]), driver='GeoJSON')
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
        data[s['properties']['Delta']] = sorted(s['properties']['basins'])

    with open(str(target[0]), 'w') as outfile:
        json.dump(data, outfile)
    return 0


def raster_pixel_areas(env, target, source):
    with rasterio.open(str(source[0]), 'r') as rast:
        shape = rast.shape
        rastcrs = rast.crs
        affine = rast.affine
        kwargs = rast.meta
    del kwargs['transform']
    areas = np.zeros(shape, dtype=np.float)

    if rastcrs['init'] == 'epsg:4326':
        geod = ccrs.PlateCarree().as_geodetic()
    else:
        raise NotImplementedError, 'Only works with lat/lon grid, assumes pixels rows have constant area'

    lon0, lat0 = affine * (0,0)
    lon1, lat1 = affine * shape[::-1]
    dlon_2 = affine.a / 2.
    dlat = affine.e
    dlat_2 = dlat / 2.
    lats = np.arange(lat0, lat1, dlat)
    for j, lat in enumerate(lats):
        lat_next = lat + dlat
        laea = ccrs.LambertAzimuthalEqualArea(central_longitude=0,
                                              central_latitude=lat+dlat_2)
        poly = sgeom.Polygon(
                laea.transform_points(geod,
                                      np.array([-dlon_2, -dlon_2, dlon_2, dlon_2]),
                                      np.array([lat, lat_next, lat_next, lat])))
        areas[j,:] = poly.area / 1e6

    with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
        dst.write(areas, 1)
