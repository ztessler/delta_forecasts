import os
import datetime
import numpy as np
import scipy
import pandas
import geopandas
import cartopy.crs as ccrs
from netCDF4 import Dataset


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


def extract_future_delta_discharge(env, source, target):
    mouths = pandas.read_pickle(str(source[0]))
    year = env['year']
    nc = Dataset(str(source[1]), 'r')
    dates = [datetime.datetime(year, 1, 1) + datetime.timedelta(days=float(d)) for d in nc.variables['time'][:]]
    dis = pandas.DataFrame(index=dates, columns=mouths.index)
    for (delta, basinid), (x, y, _, _) in mouths.iterrows():
        dis.loc[:, (delta, basinid)] = nc.variables['discharge'][:, -y-1, x] # inverted y-axis
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


    # env.Command(
            # source=config['dis_future'],
            # target=config['dis_future_extremes'],
            # action=ph.model_extremes,
            # thresh=0.99,
            # return_period=30,
            # window=30)
def model_extremes(env, source, target):
    from scipy.stats import genpareto
    thresh = float(env['thresh'])
    return_period = float(env['return_period']) #years
    window_len = int(env['window']) #years

    dis = pandas.read_pickle(str(source[0]))
    year0 = dis.index[0].strftime('%Y')
    year1 = dis.index[-1].strftime('%Y')
    window0_year_end = str(int(year0)+(window_len-1))
    window1_year_start = str(int(year1)-(window_len-1))
    windows = [dis[:window0_year_end].index[[0, -1]],
               dis[window1_year_start:].index[[0, -1]]]
    window_names = [' to '.join(map(lambda s: s.strftime('%Y'), window)) for window in windows]
    rcps = dis.columns.get_level_values('RCP').unique(),

    extremes = pandas.DataFrame(
            index=dis.columns.droplevel('RCP'),
            columns=pandas.MultiIndex.from_product([rcps, window_names]))

    plu = percentile/100.
    pgu = 1 - plu
    for (delta, basinid), rcp_time_data in list(extremes.iterrows())[:5]:
        for rcp in rcps:
            for window, winname in zip(windows, window_names):
                d = dis.loc[window[0]:window[1], (rcp, delta, basinid)]
                u = np.percentile(w, 100*plu)
                d0 = d[d>u] - u
                fit = genpareto.fit(d0)
                return_val = u + genpareto.ppf((1-(1./(return_period*365))-plu)/pgu, *fit)
                zscore = (return_val - d.mean()) / d.std(ddof=1)
                extremes.loc[(delta, basinid), (rcp, winname)] = zscore
    extremes.to_pickle(str(target[0]))
    return 0

