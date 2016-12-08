import os
import json
import csv
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import pint
import pandas
import geopandas
from affine import Affine
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, RESAMPLING
from rasterstats import zonal_stats
import shapely.geometry as sgeom
import cartopy.crs as ccrs


def clip_pop_to_delta(env, target, source):
    delta = geopandas.read_file(str(source[0]))

    with rasterio.open(str(source[1]), 'r') as src:
        kwargs = src.meta.copy()
        del kwargs['transform']

        mask = rasterize(delta.loc[0, 'geometry'], default_value=1, fill=0, out_shape=src.shape, transform=src.affine, dtype=src.dtypes[0])
        window = rasterio.get_data_window(mask, 0)
        image = src.read(1, window=window)
        mask = mask[slice(*window[0]), slice(*window[1])]
        image[mask==0] = src.nodata

        kwargs.update({
            'height': window[0][1] - window[0][0],
            'width': window[1][1] - window[1][0],
            'affine': src.window_transform(window)})

        with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
            dst.write(image, 1)

        return 0

def pop_elevation_bins(env, target, source):
    delta = geopandas.read_file(str(source[0]))
    lon, lat = np.array(delta.centroid.squeeze())
    minlon, minlat, maxlon, maxlat = np.array(delta.bounds.squeeze())

    laea = ccrs.LambertAzimuthalEqualArea(central_longitude=lon,
                                          central_latitude=lat)
    area_sqkm = delta.to_crs(laea.proj4_params)['geometry'].area.squeeze() / 1e6

    with rasterio.open(str(source[1]), 'r') as srtm_fd:
        srtm_raw = srtm_fd.read(1)
        srtm_raw_crs = srtm_fd.crs
        srtm_raw_affine = srtm_fd.affine
        srtm_raw_width = srtm_fd.width
        srtm_raw_height = srtm_fd.height
        srtm_raw_nodata = srtm_fd.nodata

    with rasterio.open(str(source[2]), 'r') as pop_fd:
        kwargs = pop_fd.meta.copy()
        pop_raw = pop_fd.read(1)
        pop_raw_crs = pop_fd.crs
        pop_raw_affine = pop_fd.affine
        pop_raw_bounds = pop_fd.bounds
        pop_raw_width = pop_fd.width
        pop_raw_height = pop_fd.height
        pop_raw_nodata = pop_fd.nodata

    # estimate reprojection params and pixel sizes based on population grid
    dst_crs = laea.proj4_params
    dst_affine, dst_width, dst_height = calculate_default_transform(
            pop_raw_crs, dst_crs, pop_raw_width, pop_raw_height,
            *pop_raw_bounds)

    pop = np.ones((dst_height, dst_width), dtype=rasterio.float64)
    srtm = np.ones((dst_height, dst_width), dtype=rasterio.float64)

    reproject(pop_raw, pop, pop_raw_affine, pop_raw_crs, pop_raw_nodata,
            dst_affine, dst_crs, pop_raw_nodata, RESAMPLING.bilinear)
    reproject(srtm_raw, srtm, srtm_raw_affine, srtm_raw_crs, srtm_raw_nodata,
            dst_affine, dst_crs, srtm_raw_nodata, RESAMPLING.bilinear)

    good = np.logical_and(pop != pop_raw_nodata, srtm != srtm_raw_nodata)
    pops = pandas.Series(name='Population', index=pandas.Index([], dtype='float'))
    elevs = np.r_[np.arange(35+1, dtype='float'), np.inf]
    for elev in elevs:
        under = np.logical_and(good, srtm <= elev)
        over = np.logical_and(good, srtm > elev)
        frac_under = under.sum() / float(good.sum())
        pops[elev] = pop[under].mean() * frac_under * area_sqkm
    pops.sort_index().to_pickle(str(target[0]))
    return 0


def group_delta_pop_elevations(env, target, source):
    deltas = env['deltas']
    delta_pop_series = OrderedDict()
    for delta, dfile in zip(deltas, source):
        delta_pop_series[delta] = pandas.read_pickle(str(dfile))
    pops = pandas.DataFrame.from_dict(delta_pop_series, orient='columns')
    pops.to_pickle(str(target[0]))
    return 0


def plot_hypsometric(env, target, source):
    pops = pandas.read_pickle(str(source[0]))
    tot_pop = pops.loc[np.inf]
    pops = pops.loc[pops.index.drop(np.inf)]
    plt.style.use('ggplot')
    f, a = plt.subplots(1, 1)
    pops.plot(ax=a, title=env['delta'])
    color = a.lines[0].get_color()

    trans = mpl.transforms.blended_transform_factory(
                a.transAxes, a.transData)
    a.scatter([1.03], [tot_pop], c=color, s=40, clip_on=False, zorder=10, transform=trans)

    a.set_xlabel('Elevation, m')
    a.set_ylabel('Population at or below elevation')
    f.savefig(str(target[0]))
    plt.close(f)
    return 0


def forecast_un_pop_elev(env, target, source):
    delta_pops = pandas.read_pickle(str(source[0]))
    with open(str(source[1]), 'r') as fin:
        countries = json.load(fin)
    popdata = pandas.read_excel(str(source[2]), sheetname=None)
    popyear = env['popyear']
    forecasts = env['forecasts']
    scenarios = ['estimates'] + env['pop_scenario_names']

    def clean_df(df):
        def clean_colname(s):
            try:
                c = int(s)
            except ValueError:
                c = str(s)
            return c
        df.columns = map(clean_colname, df.iloc[14,:])
        df = df.iloc[15:,:]
        df = df.drop([u'Index', u'Variant'], axis=1).set_index(u'Country code')
        return df

    futurepop = {}
    sheet_scenarios = ['ESTIMATES', 'LOW VARIANT', 'MEDIUM VARIANT', 'HIGH VARIANT']
    for sheet_scenario, scenario in zip(sheet_scenarios, scenarios):
        futurepop[scenario] = clean_df(popdata[sheet_scenario])

    scenarios = pandas.CategoricalIndex(scenarios[1:], categories=scenarios[1:], ordered=True)
    multiindex = pandas.MultiIndex.from_product([delta_pops.columns, forecasts, scenarios],
                                                 names=['Delta','Forecast','Pop_Scenario'])
    pop_elevs = pandas.DataFrame(index=delta_pops.index, columns=multiindex, dtype='float')
    for delta, popelevs in delta_pops.iteritems():
        for scenario in scenarios:
            for forecast in forecasts:
                growth = 0.0
                for country, cdata in countries[delta].iteritems():
                    iso = int(cdata['iso_num'])
                    if popyear in futurepop['estimates']:
                        cur_pop = futurepop['estimates'][popyear][iso]
                    else:
                        cur_pop = futurepop[scenarios[2]][popyear][iso] # scenarios[2] is "medium"
                    growth += futurepop[scenario][forecast][iso] / cur_pop * cdata['area_frac']
                pop_elevs[delta, forecast, scenario] = popelevs * growth
    pop_elevs.to_pickle(str(target[0]))
    return 0


def forecast_ssp_pop_elev(env, source, target):
    hypso = pandas.read_pickle(str(source[0]))
    # ssp = env['ssp']
    ssp_pops = pandas.read_pickle(str(source[1]))
    scenarios = env['scenarios']
    forecasts = ssp_pops.columns.levels[ssp_pops.columns.names.index('Forecast')]
    refyear = env['refyear']
    refssp = env['refssp']
    ref_ssp_pops = ssp_pops.xs(refssp, axis=1, level='SSP')

    newyears = ref_ssp_pops.columns.insert(0, refyear).sort_values()
    refpops = ref_ssp_pops.reindex(columns=newyears).interpolate(axis=1)[refyear]
    growth_factors = ssp_pops.div(refpops, axis=0)

    # scenarios = pandas.CategoricalIndex(scenarios, categories=scenarios, ordered=True)
    multiindex = pandas.MultiIndex.from_product([hypso.columns, forecasts, scenarios],
                                                 names=['Delta','Forecast','Pop_Scenario'])
    pop_elevs = pandas.DataFrame(index=hypso.index, columns=multiindex, dtype='float')
    for delta, popelevs in hypso.iteritems():
        for scenario in scenarios:
            for forecast in forecasts:
                pop_elevs[delta, forecast, scenario] = popelevs * growth_factors.loc['Mekong', (scenario, forecast)]
    pop_elevs.to_pickle(str(target[0]))
    return 0


def adjust_hypso_for_rslr(env, source, target):
    Q_ = pint.UnitRegistry().Quantity

    pop_elevs = pandas.read_pickle(str(source[0]))
    rslrs = pandas.read_pickle(str(source[1]))
    elevyear = env['elevyear']

    target_elevs = pop_elevs.index.tolist()
    target_elevs.remove(np.inf)
    target_elevs = np.array(target_elevs)
    adj_pop = pandas.DataFrame(index=target_elevs, columns=pop_elevs.columns, dtype='float')

    for (delta, forecast), _ in pop_elevs.groupby(level=['Delta', 'Forecast'], axis=1):
        years = Q_(forecast - elevyear, 'year')
        rslr = Q_(rslrs[delta], 'mm/year')
        rise = rslr * years

        new_elevs = (Q_(target_elevs, 'm') - rise).to('m').magnitude
        all_elevs = np.sort(list(set(np.r_[new_elevs, np.arange(np.max(target_elevs)+1)])))

        pops = pop_elevs[delta, forecast].loc[target_elevs]
        pops.index = new_elevs # old values but now at adjusted elevations
        pops = pops.reindex(all_elevs) # add original elevations back into index (0,1,2,...) now with nans
        pops = pops.interpolate('spline', order=3)
        pops = pops.reindex(target_elevs) # drop new_elevs, keeping only whole elevations (0,1,2,...)

        adj_pop[delta, forecast] = pops

    adj_pop.to_pickle(str(target[0]))
    return 0


def rasterize_ssp_data(env, source, target):
    with rasterio.open(str(source[0]), 'r') as basins_rast:
        basins = basins_rast.read(1)
        profile = basins_rast.profile.copy()
    affine = profile['affine']
    basinids = pandas.read_pickle(str(source[1]))
    ssp_years = env['ssp_years']
    scaling = env['scaling']
    orig_data_shape = (180*2, 360*2) #.5 x .5 degree raster
    orig_data_lats = np.linspace(90, -90, orig_data_shape[0], endpoint=False)
    orig_affine = Affine(0.5, 0, -180, 0, -0.5, 90)
    dy = -.5
    pc = ccrs.PlateCarree()
    orig_data_areas = []
    for lat in orig_data_lats:
        laea = ccrs.LambertAzimuthalEqualArea(0.25, lat+dy/2.)
        p = sgeom.Polygon([(0, lat), (0, lat+dy), (.5, lat+dy), (.5, lat), (0, lat)])
        ps = laea.project_geometry(p, src_crs=pc)
        orig_data_areas.append(ps.area / (1000**2)) # m2 to km2

    data = np.ones(basins.shape + (len(ssp_years),)) * -1
    to_floor_int = lambda s: int(np.floor(s))
    with open(str(source[2]), 'r') as csvfile:
        ssp_data = csv.reader(csvfile)
        next(ssp_data)
        for pixel in ssp_data:
            lon_lat = (float(pixel[1]), float(pixel[2]))
            x, y = map(to_floor_int, ~affine * lon_lat) # lon, lat
            _, orig_y = map(to_floor_int, ~orig_affine * lon_lat)
            data[y, x, :] = np.array(map(float, pixel[4:])) * scaling / orig_data_areas[orig_y] # convert from millions of people to people/sqkm (or $)
    mask = (data == -1)
    indices = distance_transform_edt(mask[...,0], return_distances=False, return_indices=True)
    data = data[tuple(indices)]
    data[basins==profile['nodata']] = profile['nodata']

    profile.update(count=len(ssp_years))
    with rasterio.open(str(target[0]), 'w', **profile) as out:
        out.write(np.rollaxis(data, 2))
    return 0


def extract_delta_ssp_vals(env, source, target):
    deltas = geopandas.read_file(str(source[0])).set_index('Delta')
    ssps = {}
    for i, ssp in enumerate(env['ssp_names']):
        with rasterio.open(str(source[1+i]), 'r') as ssp_rast:
            ssps[ssp] = ssp_rast.read()
            profile = ssp_rast.profile.copy()
            ssp_raw_crs = ssp_rast.crs
            ssp_raw_affine = ssp_rast.affine
            ssp_raw_bounds = ssp_rast.bounds
            ssp_raw_width = ssp_rast.width
            ssp_raw_height = ssp_rast.height
            ssp_raw_nodata = ssp_rast.nodata
    multiindex = pandas.MultiIndex.from_product([env['ssp_names'], env['ssp_years']],
                                                names=['SSP', 'Forecast'])
    data = pandas.DataFrame(index=deltas.index, columns=multiindex, dtype='float')
    # import ipdb;ipdb.set_trace()
    for delta in deltas.index:
        lon, lat = np.array(deltas.centroid[delta])
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=lon, central_latitude=lat)
        delta_proj = deltas.loc[[delta]].to_crs(proj.proj4_params)
        bounds_proj = delta_proj.buffer(50000).bounds.squeeze()
        bounds = transform_bounds(proj.proj4_params, ssp_raw_crs, *bounds_proj)
        lon1, lat1, lon2, lat2 = bounds
        x1, y1 = map(lambda x: int(np.floor(x)), ~ssp_raw_affine * (lon1, lat1))
        x2, y2 = map(lambda x: int(np.ceil(x)), ~ssp_raw_affine * (lon2, lat2))

        dst_affine, dst_width, dst_height = calculate_default_transform(
                ssp_raw_crs, proj.proj4_params, x2-x1, y1-y2,
                *bounds)
        for i, ssp in enumerate(env['ssp_names']):
            for j, forecast in enumerate(env['ssp_years']):
                delta_ssp = ssps[ssp][j,y2:y1,x1:x2] # lats inverted
                ssp_proj = np.ones((dst_height, dst_width), dtype=rasterio.float64)
                reproject(ssps[ssp][j,...], ssp_proj, ssp_raw_affine, ssp_raw_crs, ssp_raw_nodata,
                    dst_affine, proj.proj4_params, ssp_raw_nodata, RESAMPLING.bilinear)
                stats = zonal_stats(delta_proj.loc[delta,'geometry'], ssp_proj, affine=dst_affine, nodata=ssp_raw_nodata)
                counts = 0
                means = 0
                for feat in stats:
                    if feat['count'] > 0:
                        counts += feat['count']
                        means += feat['mean'] * feat['count']
                if counts > 0:
                    deltamean = float(means)/float(counts)
                else:
                    deltamean = np.nan
                data.loc[delta, (ssp, forecast)] = deltamean * delta_proj.area.squeeze()/(1000**2)
    data.to_pickle(str(target[0]))
    return 0
