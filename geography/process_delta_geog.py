import csv
import numpy as np
from scipy.ndimage import distance_transform_edt
import json
import pandas
import geopandas
import fiona
import rasterio
from rasterio.features import rasterize
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


def locate_basin_mouths(env, target, source):
    basin_info = pandas.read_pickle(str(source[0]))
    with rasterio.open(str(source[1]), 'r') as rast:
        affine = rast.affine
    with open(str(source[2]), 'r') as f:
        basin_ids = json.load(f)

    delta_basins = []
    for delta, basins in basin_ids.iteritems():
        delta_basins.extend([(delta, basin) for basin in basins])
    index = pandas.MultiIndex.from_tuples(delta_basins)

    mouths = pandas.DataFrame(index=index, columns=['x', 'y',
                                                    'lon_center',
                                                    'lat_center'])
    for delta, basin in delta_basins:
        lon = basin_info.loc[basin]['MouthXCoord']
        lat = basin_info.loc[basin]['MouthYCoord']
        x, y = ~affine * (lon, lat)

        mouths.loc[delta, basin]['lon_center'] = lon
        mouths.loc[delta, basin]['lat_center'] = lat
        mouths.loc[delta, basin]['x'] = int(x)
        mouths.loc[delta, basin]['y'] = int(y)

    mouths.to_pickle(str(target[0]))
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


# def _get_clean_iso_alpha3(prop):
    # isoa3 = prop['iso_alpha3']
    # if isoa3 is None:
        # if prop['NAME'] == 'In dispute SOUTH SUDAN/SUDAN':
            # return 'SSD' # SOUTH SUDAN
    # return prop['iso_alpha3']
def rasterize_gnp(env, target, source):
    def most_recent_data(c):
        year = 0
        gdp = None
        for k, v in c.iteritems():
            try:
                if int(k) > year and v: # no data years have empty string values
                    year = int(k)
                    gdp = float(v)
            except ValueError:
                pass
        return gdp

    with rasterio.open(str(source[2]), 'r') as rast: # basin raster for grid geometry
        mask = rast.read_masks(1)
        meta = rast.meta.copy()
    del meta['transform']

    with open(str(source[1]), 'r') as fd: # gnp data in csv
        fd.readline()
        fd.readline()
        fd.readline()
        fd.readline()
        reader = csv.DictReader(fd)
        gdps = {}
        for country in reader:
            gdps[country['Country Code']] = most_recent_data(country)
    gdps = {c: gdp if gdp is not None else meta['nodata'] for (c,gdp) in gdps.iteritems()}

    with rasterio.open(str(target[0]), 'w', **meta) as rast:
        with fiona.open(str(source[0]), 'r') as countries: # country boudaries
            assert countries.crs == meta['crs']
            gdp = rasterize(
                    shapes=((feat['geometry'], gdps.get(feat['properties']['iso_alpha3'], meta['nodata']))
                                for feat in countries),
                    out_shape=(meta['height'], meta['width']),
                    fill=meta['nodata'],
                    transform=meta['affine'],
                    dtype=meta['dtype'],
                    )

        # fill missing data and reset ocean mask
        nearest = distance_transform_edt(gdp==meta['nodata'], return_distances=False, return_indices=True)
        gdp = gdp[tuple(nearest)]
        gdp[~mask.astype(bool)] = meta['nodata']
        rast.write(gdp, 1)
