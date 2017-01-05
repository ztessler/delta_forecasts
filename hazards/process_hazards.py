import os
import datetime
import re
import json
import numpy as np
import scipy
import pandas
import geopandas
import itertools
from collections import defaultdict
from rasterio.features import rasterize
import cartopy.crs as ccrs
from netCDF4 import Dataset
import requests
from affine import Affine
from bs4 import BeautifulSoup
import time

from util import in_new_process



def storm_surge_agg_points(env, target, source):
    deltas = geopandas.read_file(str(source[0])).set_index('Delta')
    allsurges = geopandas.read_file(str(source[1]))

    origcols = ['T_10', 'T_25', 'T_50', 'T_100', 'T_250']
    cols = map(lambda s: float(s[2:]), origcols)
    allsurges = allsurges[['geometry'] + origcols].rename_axis(
            {orig:new for orig, new in zip(['geometry']+origcols, ['geometry']+cols)},
            axis=1)

    centroids = deltas.centroid
    mean_surge = pandas.DataFrame(index=deltas.index, columns=cols, dtype='float')
    for dname in deltas.index:
        lon, lat = np.array(centroids.loc[dname])
        # reproject delta shape to Azimuthal Equidistant - distances are correct from center point, good for buffering
        aed = ccrs.AzimuthalEquidistant(central_longitude=lon,
                                        central_latitude=lat)
        delta = deltas.loc[[dname]].to_crs(aed.proj4_params)['geometry']
        # buffer around convex hull (very slow over all points)
        delta_buff = delta.convex_hull.buffer(25 * 1000) # 25km
        # project back to match surge data
        poly = delta_buff.to_crs(allsurges.crs).item()

        surges = allsurges[allsurges.within(poly)][cols]

        # 0 seems to be used as a flag for missing data
        # very small surges also in data, remove these
        # negative surge also clearly wrong
        surges[surges <= 0.01] = np.nan
        # remove where there is no or negative increase between levels. larger surges are by defitition rarer
        surges[surges.diff(axis=1) <= 0] = np.nan
        # remove any values at higher levels from bad values
        for i, col in enumerate(surges):
            for j in np.where(surges[col].isnull())[0]:
                for ii in range(i+1, len(cols)):
                    surges.iloc[j, ii] = np.nan

        # surges = surges.dropna(axis=0, how='any')
        # for i, level in enumerate(surges):
            # surges[surges[level].isnull()].iloc[:, i:] = np.nan

        # now all cleaned, each row has sane values
        # starts with positive value, increases from there
        # can't take mean across all existing data, mean can end up with decreasing values
        # instead, take mean over rows that have the most available data
        dropcols = surges.isnull().sum(axis=1).min()
        if dropcols > 0:
            surges = surges[cols[:-dropcols]]
        surges = surges.dropna(how='any', axis=0)

        agg_surge = surges.mean(axis=0)
        assert (agg_surge.diff().dropna()>0).all(), 'Processing should enforce increasing surge at longer return periods, went wrong'
        mean_surge.loc[dname,:] = agg_surge

    mean_surge.to_pickle(str(target[0]))
    return 0


def storm_surge_populations(env, source, target):
    surge = pandas.read_pickle(str(source[0])).astype(float)
    populations = pandas.read_pickle(str(source[1]))

    # columns is multiindex, (delta, rslr/pop forecast year, pop_scenario)
    # index is return period
    # value is people exposed
    exposure = pandas.DataFrame(index=surge.columns, columns=populations.columns, dtype='float')

    # for (delta, forecast), _ in pop_elevs.groupby(level=['delta', 'forecast'], axis=1):
    for (delta, forecast, pop_scenario), pop in populations.iteritems():
        spline = scipy.interpolate.InterpolatedUnivariateSpline(np.array(pop.index), pop.values)
        exposure.loc[:,(delta, forecast, pop_scenario)] = spline(np.array(surge.loc[delta,:]))

    exposure.to_pickle(str(target[0]))
    return 0


def surge_expected_expo(env, target, source):
    '''
    Compute expected flooding due to surge for any given year.   Adjusts to avoid double counting
    Estimated as:
        (prob of 25 year flood) * (exposure to 25 year flood) +
        (prob of 10 year flood - prob of 25 year flood) * (exposure to 10 year flood) +
        ...
    '''
    exposure = pandas.read_pickle(str(source[0]))

    def one_yr_expected(s):
        total_exp = 0.0
        prob_more_extreme = 0.0
        for ret_period, ret_level in s.sort_index(ascending=False).iteritems():
            if ret_level is np.nan:
                continue
            prob = 1./ret_period
            total_exp += (prob - prob_more_extreme) * ret_level
            prob_more_extreme += prob
        return total_exp

    exposure.apply(one_yr_expected, axis=0).to_pickle(str(target[0]))
    return 0


@in_new_process
def extract_future_delta_discharge(env, source, target):
    mouths = pandas.read_pickle(str(source[0]))
    year = env['year']
    nc = Dataset(str(source[1]), 'r')
    dates = [datetime.datetime(year, 1, 1) + datetime.timedelta(days=float(d)) for d in nc.variables['time'][:]]
    disgrid = nc.variables['discharge'][:]
    dis = pandas.DataFrame(
            disgrid[:, -mouths['y']-1, mouths['x']],
            index=dates,
            columns=mouths.index)
    nc.close()
    dis.to_pickle(str(target[0]))
    return 0


def combine_future_dis_years(env, source, target):
    years = env['years']
    dis = []
    for year, disfile in zip(years, source):
        dis.append(pandas.read_pickle(str(disfile)))
    fulldis = pandas.concat(dis, axis=0)
    fulldis.to_pickle(str(target[0]))
    return 0


def combine_future_dis_rcps(env, source, target):
    rcps = [pandas.read_pickle(str(s)) for s in source]
    names = map(lambda s: 'RCP'+s, env['rcpnames'])

    dis = pandas.concat(rcps, axis=1, keys=names, names=['RCP'])
    dis.to_pickle(str(target[0]))
    return 0


def model_extremes(env, source, target):
    from scipy.stats import genpareto
    percentile = float(env['percentile'])
    return_period = float(env['return_period']) #years
    try:
        window_len = int(env['window']) #years
    except ValueError:
        if env['window'] == 'none':
            window_len = None
        else:
            raise

    historical = pandas.read_pickle(str(source[0])).astype('float64')
    if len(source) == 1:
        dis = pandas.read_pickle(str(source[0])).astype('float64')
    else:
        dis = pandas.read_pickle(str(source[1])).astype('float64')
    year0 = dis.index[0].strftime('%Y')
    year1 = dis.index[-1].strftime('%Y')
    if window_len is None:
        windows = [dis.index[[0, -1]]]
    else:
        window0_year_end = str(int(year0)+(window_len-1))
        window1_year_start = str(int(year1)-(window_len-1))
        windows = [dis[:window0_year_end].index[[0, -1]],
                   dis[window1_year_start:].index[[0, -1]]]
    window_names = [' to '.join(map(lambda s: s.strftime('%Y'), window)) for window in windows]
    window_dict = dict(zip(window_names, windows))
    rcps = dis.columns.get_level_values('RCP').unique().tolist()

    deltas_basins = dis.columns.droplevel('RCP').unique()
    d_b_r_w = [tuple(db)+(r, w) for db in deltas_basins for r in rcps for w in window_names]
    index = pandas.MultiIndex.from_tuples(d_b_r_w,
                        names=['Delta', 'BasinID', 'RCP', 'Window'])
    extremes = pandas.DataFrame(
            index=index,
            columns=['zscore', 'mean', 'std'],
            dtype=np.float64)

    plu = percentile/100.
    pgu = 1 - plu
    for (delta, basinid, rcp, winname), rcp_time_data in list(extremes.iterrows()):
        window = window_dict[winname]
        d = dis.loc[window[0]:window[1], (rcp, delta, basinid)]
        d0 = historical.loc[:, ('RCPnone', delta, basinid)]
        u = np.percentile(d, 100*plu)
        dtail = d[d>u] - u
        fit = genpareto.fit(dtail)
        return_val = u + genpareto.ppf((1-(1./(return_period*365))-plu)/pgu, *fit)
        zscore = (return_val - d0.mean()) / d0.std(ddof=1)
        mean = d.mean()
        std = d.std(ddof=1)
        extremes.loc[(delta, basinid, rcp, winname)] = (zscore, mean, std)
    extremes.to_pickle(str(target[0]))
    return 0


def agg_delta_extremes(env, source, target):
    extremes = pandas.read_pickle(str(source[0]))
    newindex = extremes.index.droplevel('BasinID')
    agg_extremes = pandas.Series(
            index=pandas.MultiIndex.from_tuples(newindex.unique(), names=newindex.names))

    bydelta = extremes.groupby(level='Delta')
    for delta, deltadata in bydelta:
        runs = deltadata.groupby(level=['RCP', 'Window'])
        for (rcp, winname), rundata in runs:
            zscore_weighted = (rundata['zscore'] * rundata['mean'] / rundata['mean'].sum()).sum()
            agg_extremes.loc[(delta, rcp, winname)] = zscore_weighted

    agg_extremes.to_pickle(str(target[0]))
    return 0


def concat_hist_fut_extremes(env, source, target):
    hist = pandas.read_pickle(str(source[0]))
    fut = pandas.read_pickle(str(source[1]))
    extremes = pandas.concat([hist, fut], axis=0).sortlevel(1).sortlevel(0)
    extremes.to_pickle(str(target[0]))
    return 0


def get_waves_nclist(env, source, target):
    urlformat = env['urlformat']
    url = os.path.join(os.path.dirname(urlformat), 'catalog.html')
    isnc = re.compile(r'catalog.html\?dataset=.*_(\d{6})\.nc')
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    with open(str(target[0]), 'w') as fout:
        for link in soup.find_all('a'):
            match = re.match(isnc, link.get('href'))
            if match:
                fout.write(match.group(1) + '\n')
    return 0


def waves_find_delta_indices(env, source, target):
    deltas = geopandas.read_file(str(source[0])).set_index('Delta')
    url = env['url']
    nullval = env['nullval']
    nc = Dataset(url)
    assert nc.variables['y'].long_name == 'Latitude'
    assert nc.variables['x'].long_name == 'Longitude'
    pc = ccrs.PlateCarree()
    hs = nc.variables['Hs'][0,:,:]
    nt, ny, nx = nc.variables['Hs'].shape
    miny, maxy = nc.variables['y'][[0, -1]]
    minx, maxx = nc.variables['x'][[0, -1]]
    dx = (maxx - minx + 1.) / nx
    dy = (maxy - miny + 1.) / ny
    affine = Affine.from_gdal(minx, dx, 0, miny, 0, dy)

    centroids = deltas.centroid
    indices = defaultdict(dict)

    for dname in deltas.index:
        lon, lat = np.array(centroids.loc[dname])
        # reproject delta shape to Azimuthal Equidistant - distances are correct from center point, good for buffering
        aed = ccrs.AzimuthalEquidistant(central_longitude=lon,
                                        central_latitude=lat)
        delta = deltas.loc[[dname]].to_crs(aed.proj4_params)['geometry']
        npixels = 0
        for buff_km in itertools.chain([10, 25], itertools.count(50, 50)):
            # buffer around convex hull (very slow over all points)
            delta_buff = delta.convex_hull.buffer(buff_km * 1000)
            poly = delta_buff.to_crs(pc.proj4_params).item()
            mask = rasterize([(poly, 1)], out_shape=(ny, nx), fill=0, transform=affine, all_touched=True)
            mask[hs==nullval] = 0
            delta_indices = np.where(mask==1)
            npixels = len(delta_indices[0])
            if npixels >= 3:
                break
        indices[dname]['y'] = delta_indices[0].tolist()
        indices[dname]['x'] = delta_indices[1].tolist()
        indices[dname]['buffer'] = buff_km

    with open(str(target[0]), 'w') as fout:
        json.dump(indices, fout)
    return 0


def process_wave_month(env, source, target):
    # indexing opendap files only download necessary data
    # But, netcdf doesn't do numpy "fancy" indexing to pull out specific indices
    # so can either get data one pixel at a time(v1), entire grid and then use
    # numpy fancy indexing(v2), or get surrounding block for each delta and pull
    # out pixels of interest from the diagonal(v3).
    # v1 is fastest - 1.6 minutes, vs 4.6 vs 7.2
    # since most time is spent downloading and processing in C code, can use threads
    # to do different months in parallel to speed this up substaintially
    url = env['url']
    with open(str(source[0]), 'r') as fin:
        indices = json.load(fin)
    deltas = sorted(indices.keys())
    pixels = [(delta, pixel) for delta in deltas for pixel in range(len(indices[delta]['x']))]

    nc = Dataset(url)

    times = ['{}:{:06}'.format(*tt) for tt in zip(nc.variables['time1'][:], nc.variables['time2'][:])]
    datetimes = [datetime.datetime.strptime(t, '%Y%m%d:%H%M%S') for t in times]
    timeindex = pandas.DatetimeIndex(datetimes)
    delta_pixels = pandas.MultiIndex.from_tuples(pixels, names=['Delta', 'Pixel'])
    waves = pandas.DataFrame(index=timeindex, columns=delta_pixels)

    power_fac = (1026 * 9.8**2) / (64 * np.pi)
    for delta in deltas:
        ys = np.array(indices[delta]['y'])
        xs = np.array(indices[delta]['x'])
        for i, (y, x) in enumerate(zip(ys, xs)): #v1 1.6 minutes
            sig_height = nc.variables['Hs'][:, y, x] #v1
            period = nc.variables['Tm'][:, y, x] #v1
            power = power_fac * sig_height**2 * period # W/m of crest #v1
            waves.loc[:, (delta, i)] = power #v1
    nc.close()
    waves.to_pickle(str(target[0]))
    return 0


def concat_waves_times(env, source, target):
    dfs = []
    fnames = sorted([str(s) for s in source]) # file names vary only in yyyymm, put them in order
    for f in fnames:
        dfs.append(pandas.read_pickle(f))
    waves = pandas.concat(dfs, axis=0)
    waves[waves<0] = 0
    waves.to_pickle(str(target[0]))
    return 0


def waves_avg_pixels(env, source, target):
    waves = pandas.read_pickle(str(source[0]))
    waves = waves.resample('1D', how='mean').dropna(axis=0) # Feb 29 gets np.nan on non-leap-years
    npixels = 3
    largestpixels = waves.mean(axis=0).groupby(level='Delta').apply(
            lambda s: np.argpartition(-s, npixels-1)[:npixels])
    dwaves = waves.groupby(level='Delta', axis=1)
    meanwaves = pandas.DataFrame(
            index=waves.index,
            columns=sorted(waves.columns.droplevel(level='Pixel').unique())
            )
    for dname, deltapixels in dwaves:
        meanwaves.loc[:, dname] = waves.loc[:, zip(itertools.repeat(dname), largestpixels[dname])].mean(axis=1)
    meanwaves.to_pickle(str(target[0]))
    return 0


def compute_waves_extremes(env, source, target):
    from scipy.stats import genpareto
    percentile = float(env['percentile'])
    return_period = float(env['return_period']) #years

    historical = pandas.read_pickle(str(source[0])).astype('float64')
    if len(source) == 1:
        waves = pandas.read_pickle(str(source[0])).astype('float64')
    else:
        waves = pandas.read_pickle(str(source[1])).astype('float64')

    extremes = pandas.DataFrame(
            index=waves.columns,
            columns=['zscore', 'mean', 'std'],
            dtype=np.float64)

    plu = percentile/100.
    pgu = 1 - plu
    for delta, w in waves.iteritems():
        u = np.percentile(w, 100*plu)
        wtail = w[w>u] - u
        fit = genpareto.fit(wtail)
        return_val = u + genpareto.ppf((1-(1./(return_period*365))-plu)/pgu, *fit)
        w0 = historical.loc[:, delta]
        zscore = (return_val - w0.mean()) / w0.std(ddof=1)
        mean = w.mean()
        std = w.std(ddof=1)
        extremes.loc[delta] = (zscore, mean, std)
    extremes.to_pickle(str(target[0]))
    return 0


def agg_wave_zscores(env, source, target):
    dfs = []
    for s in source:
        dfs.append(pandas.read_pickle(str(s)))
    new_level_names = env['level_names']
    if not isinstance(new_level_names, list):
        new_level_names = [new_level_names]
    level_names = new_level_names + dfs[0].columns.names
    levels = env['levels']
    waves = pandas.concat(dfs, axis=1, keys=levels, names=level_names)
    waves.to_pickle(str(target[0]))
    return 0


def clean_wave_zscores(env, source, target):
    waves = pandas.read_pickle(str(source[0]))
    stacked = waves.stack(level=['GCM', 'RCP', 'Window'])['mean']
    nogcms = stacked.unstack('GCM').mean(axis=1)
    nogcms.to_pickle(str(target[0]))
    return 0


def combine_hazard_scores(env, source, target):
    dis = pandas.read_pickle(str(source[0]))
    waves = pandas.read_pickle(str(source[1]))

    common_names = {
            # RCPs
            'RCP2p6': 'low',
            'RCP4.5': 'low',

            'RCP8p5': 'high',
            'RCP8.5': 'high',

            # Windows
            '2006 to 2035': 'early',
            'MID21C': 'early',

            '2070 to 2099': 'late',
            'END21C': 'late',
            }

    dis.rename(common_names, inplace=True)
    waves.rename(common_names, inplace=True)

    hazards = pandas.concat({'Discharge': dis, 'Waves': waves}, axis=1)
    hazards.index.names = dis.index.names
    hazards.to_pickle(str(target[0]))
    return 0

