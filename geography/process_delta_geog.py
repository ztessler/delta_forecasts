import sys
import csv
import json
import numpy as np
from scipy.ndimage import distance_transform_edt
import pandas
import geopandas
import fiona
import rasterio
from rasterio.features import rasterize
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from rasterstats import zonal_stats
from collections import OrderedDict, defaultdict
import networkx as nx
import pint

from util import in_new_process, getLogger


def clean_delta_name(delta):
    return delta.replace(' ','_').replace('-','_')


def set_delta_val(env, target, source):
    deltas = pandas.read_pickle(str(source[0]))
    series = pandas.Series(env['val'], index=deltas.index)
    series.to_pickle(str(target[0]))
    return 0


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


def calc_delta_areas(env, target, source):
    deltas = geopandas.GeoDataFrame.from_file(str(source[0])).set_index('Delta')
    crs = ccrs.AlbersEqualArea()
    areas_sqkm = deltas.to_crs(crs.proj4_params).area / (1000**2)
    areas_sqkm.to_pickle(str(target[0]))


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

    tups = []
    for delta, basins in data.iteritems():
        for basin in basins:
            tups.append((delta, basin))

    index = pandas.MultiIndex.from_tuples(tups, names=['Delta', 'BasinID'])
    df = pandas.DataFrame(index=index)
    df.to_pickle(str(target[0]))
    return 0


def locate_basin_mouths(env, target, source):
    basin_info = pandas.read_pickle(str(source[0]))
    with rasterio.open(str(source[1]), 'r') as rast:
        affine = rast.affine
    basin_ids = pandas.read_pickle(str(source[2]))

    mouths = pandas.DataFrame(index=basin_ids.index, columns=['x', 'y',
                                                              'lon_center',
                                                              'lat_center'])
    for delta, basin in basin_ids.index:
        try:
            lon = basin_info.loc[basin]['MouthXCoord']
            lat = basin_info.loc[basin]['MouthYCoord']
        except KeyError:
            # DDM30 network doesn't have MouthX/YCoord info, so working with DBCell info
            mouthcell = basin_info.loc[basin_info['BasinID']==basin].iloc[0,:] # cells along river are sorted by discharge vol, first cell is mouth
            lon = mouthcell['CellXCoord']
            lat = mouthcell['CellYCoord']
        x, y = map(int, ~affine * (lon, lat))

        mouths.loc[delta, basin]['lon_center'] = lon
        mouths.loc[delta, basin]['lat_center'] = lat
        mouths.loc[delta, basin]['x'] = x
        mouths.loc[delta, basin]['y'] = y

    mouths.to_pickle(str(target[0]))
    return 0


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


def delta_countries(env, target, source):
    # import ipdb;ipdb.set_trace()
    deltas = geopandas.GeoDataFrame.from_file(str(source[0]))
    deltas = deltas.set_index('Delta')

    lcea = ccrs.LambertCylindrical()
    deltas_lcea = deltas.to_crs(lcea.proj4_params)

    d_countries = defaultdict(lambda : defaultdict(float)) # dict of {deltaname: {country: area, country: area, ...}}
    with fiona.open(str(source[1]), 'r') as borders:
        assert borders.crs['init'] == u'epsg:4326'
        crs = ccrs.PlateCarree()
        for i, delta in enumerate(deltas.index):
            print i, delta
            sys.stdout.flush()
            intersects = borders.items(bbox=deltas['geometry'][delta].bounds)
            for fid, feat in intersects:
                country = lcea.project_geometry(sgeom.shape(feat['geometry']), src_crs=crs)
                area = deltas_lcea['geometry'][delta].intersection(country).area
                if area > 0:
                    name = feat['properties']['NAME'].title()
                    iso_num = feat['properties']['iso_num']
                    d_countries[delta][(name, iso_num)] += area

    fractions = {}
    for d, cs in d_countries.iteritems():
        fractions[d] = {}
        tot_area = 0
        for c, a in cs.iteritems():
            tot_area += a
        for c, a in cs.iteritems():
            name, iso = c
            fractions[d][name] = {}
            fractions[d][name]['iso_num'] = iso
            fractions[d][name]['area_frac'] = a/tot_area

    with open(str(target[0]), 'w') as out:
        json.dump(fractions, out)
    return 0


@in_new_process
def build_basin_river_network(env, source, target):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    with rasterio.open(str(source[0]), 'r') as rast:
        basins = rast.read(1)
    with rasterio.open(str(source[1]), 'r') as rast:
        flowdir = rast.read(1)
    mouths = pandas.read_pickle(str(source[2]))

    logger = getLogger(target)

    neighbors = {
            1: (1, 0),
            2: (1, 1),
            4: (0, 1),
            8: (-1, 1),
            16: (-1, 0),
            32: (-1, -1),
            64: (0, -1),
            128: (1, -1),
            }

    nets = pandas.Series(index=mouths.index, dtype=object)
    for delta, basinid in nets.index:
        logger.info('{0} - {1}'.format(delta, basinid))
        G = nx.DiGraph()
        for y, x in zip(*np.where(basins==basinid)):
            tocell = int(flowdir[y, x])
            G.add_node((x, y))
            if tocell > 0:
                dx, dy = neighbors[tocell]
                if basins[y+dy, x+dx] == basinid:
                    G.add_edge((x, y), (x+dx, y+dy))

        nets[(delta, basinid)] = G

    nets.to_pickle(str(target[0]))
    return 0
