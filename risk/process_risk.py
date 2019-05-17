import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import itertools
from adjustText import adjust_text


def adj_annual_exposure_for_vuln(env, target, source):
    '''
    Risk = hazard * exposure * vuln
           (return periods of surge) * (populations in each surge height) * (vuln index)
    '''
    exposure = pandas.read_pickle(str(source[0]))
    vuln = pandas.read_pickle(str(source[1]))

    # pandas auto-alignment along indexes is magic
    (exposure * vuln).to_pickle(str(target[0]))


def per_capita_exposure(env, target, source):
    exposure = pandas.read_pickle(str(source[0]))
    pops = pandas.read_pickle(str(source[1]))

    per_cap_expo = exposure / pops.loc[np.inf, :]
    per_cap_expo.to_pickle(str(target[0]))
    return 0


def plot_surge_annual_exposure(env, target, source):
    mpl.style.use('ggplot')

    exposure = pandas.read_pickle(str(source[0]))
    delta = env['delta']
    exp = env['exp']

    pops = exposure.loc[delta].unstack(level='Pop_Scenario')#.iloc[:, ::-1]
    pops.columns = pops.columns.rename('Population Growth Scenario')

    color = next(iter(mpl.rcParams['axes.prop_cycle']))['color']

    f, a = plt.subplots(1, 1, figsize=(12,8))
    pops.iloc[:,1].plot(ax=a, c=color, lw=3, legend=False) # want mid-range scenario, choosing second column only works if they're ordered.  not necessarily the case for SSPs
    a.fill_between(pops.index, pops.min(axis=1), pops.max(axis=1), color=color, alpha=.3)

    a.set_ylabel('EV(annual surge exposure), people')
    a.set_xlabel('Outlook year')
    a.set_title('{0}: Surge exposure trends ({1})'.format(delta, exp))
    f.savefig(str(target[0]))
    plt.close(f)
    return 0


def mk_plot_multiscenario_exposure(a, pops, scenarios, legend=True):
    mpl.style.use('ggplot')

    color_cycle = mpl.rcParams['axes.prop_cycle']

    dummy_fills = []
    for scenario, color in zip(scenarios, color_cycle):
        # pops[name].plot(ax=a, color=color['color'], lw=2, legend=False)
        pops[scenario, pops.columns.levels[1][1]].plot(ax=a, color=color['color'], lw=3, legend=False)
        a.fill_between(pops.index, pops[scenario].min(axis=1), pops[scenario].max(axis=1), color=color['color'], alpha=.3)
        if legend:
            dummy_fills.append(mpl.patches.Rectangle((0,0),1,1, fc=color['color'], ec='none'))
    if legend:
        a.legend(dummy_fills, scenarios, loc=2, title=pops.columns.names[0])

    return


def plot_surge_annual_exposure_multiscenario(env, target, source):
    exposures = [pandas.read_pickle(str(s)) for s in source[:env['nsources']]]
    delta = env['delta']
    scenarios = env['scenarios']

    # need level names to match for concat to work
    # pops may be "Low, Medium, High", or "SSP1, SSP2, SSP3". make the same
    # assumes three pop scenarios with low,medium,high ordering
    for e in exposures:
        if 'SSP1' in e.index.get_level_values('Pop_Scenario'):
            e.index = e.index.set_levels(['Low', 'Medium', 'High'], 'Pop_Scenario')

    pops = [e.loc[delta].unstack(level='Pop_Scenario').iloc[:, ::-1] for e in exposures]
    pops = pandas.concat(pops, keys=scenarios, axis=1)
    pops.columns.rename('Environmental Scenario', level=0, inplace=True)
    pops.columns.rename('Population Growth Scenario', level=1, inplace=True)
    pops.columns.set_levels([s.title() for s in pops.columns.levels[1]], level=1, inplace=True)

    f, a = plt.subplots(1, 1, figsize=(12,8))
    mk_plot_multiscenario_exposure(a, pops, scenarios)

    a.set_ylabel('EV(annual surge exposure), people')
    a.set_xlabel('Outlook year')
    a.set_title('{}: Surge exposure trends'.format(delta))
    f.savefig(str(target[0]))
    plt.close(f)
    return 0

# env.Command(
        # source=[[c['surge_annual_exposure'] for c in configs],
        # target=config['surge_annual_exposure_ranges'],
        # action=pr.compute_exposure_ranges,
        # scenarios=scenarios)
def compute_exposure_ranges(env, target, source):
    exposures = [pandas.read_pickle(str(s)) for s in source if os.path.exists(str(s))]

    deltas = exposures[0].index.get_level_values(level='Delta').drop_duplicates()
    ranges = pandas.DataFrame(index=deltas, columns=['min', 'max'], dtype='float')
    ranges['min'] = np.inf
    ranges['max'] = -np.inf
    for e in exposures:
        mins = e.groupby(level='Delta').min()
        maxs = e.groupby(level='Delta').max()
        ranges['min'][mins < ranges['min']] = mins
        ranges['max'][maxs > ranges['max']] = maxs

    ranges.to_pickle(str(target[0]))
    return 0


def plot_surge_annual_exposure_multiscenario_multidelta(env, target, source):
    def round_to_1digit(x):
        mult10 = float(10**np.floor(np.log10(x)))
        return np.ceil(x / mult10) * mult10
    exposures = [pandas.read_pickle(str(s)) for s in source[:2]]
    ranges = pandas.read_pickle(str(source[2]))
    okdata = pandas.Series(False, index=exposures[0].sum(level='Delta').index)
    for e in exposures:
        ok_this_scenario = np.logical_and(~e.sum(level='Delta').isnull(), e.sum(level='Delta')!=0)
        okdata = np.logical_or(okdata, ok_this_scenario)
    deltas = exposures[0].sum(level='Delta').index[okdata]
    scenarios = env['scenarios']

    # need level names to match for concat to work
    # pops may be "Low, Medium, High", or "SSP1, SSP2, SSP3". make the same
    # assumes three pop scenarios with low,medium,high ordering
    for e in exposures:
        if 'SSP1' in e.index.get_level_values('Pop_Scenario'):
            e.index = e.index.set_levels(['Low', 'Medium', 'High'], 'Pop_Scenario')

    ncols = int(np.ceil(np.sqrt(len(deltas)))) + 1
    nrows = int(np.ceil(len(deltas)/float(ncols)))
    f, axs = plt.subplots(nrows, ncols, figsize=(14,8))
    for a in axs.flat:
        a.set_visible(False)

    for i, (delta, a) in enumerate(zip(deltas, axs.flat)):
        a.set_visible(True)
        pops = [e.loc[delta].unstack(level='Pop_Scenario').iloc[:, ::-1] for e in exposures]
        # minpop = np.min([p.min().min() for p in pops])
        # maxpop = np.max([p.max().max() for p in pops])
        pops = pandas.concat(pops, keys=scenarios, axis=1)
        pops.columns.rename('Environmental Scenario', level=0, inplace=True)
        pops.columns.rename('Population Growth Scenario', level=1, inplace=True)
        pops.columns.set_levels([s.title() for s in pops.columns.levels[1]], level=1, inplace=True)
        mk_plot_multiscenario_exposure(a, pops, scenarios, legend=False)

        # a.set_ylim(ranges.loc[delta, 'min'], ranges.loc[delta, 'max'])
        year0 = pops.index.get_level_values(level='Forecast').min()
        year1 = pops.index.get_level_values(level='Forecast').max()
        # a.set_ylim(0, round_to_1digit(ranges.loc[delta, 'max']))
        ymax = round_to_1digit(pops.max().max())
        a.set_ylim(0, ymax)
        print delta, ranges.loc[delta, 'max'], a.get_ylim()
        a.yaxis.set_ticks([0, ymax])
        if i == len(deltas)-1:
            a.xaxis.set_ticks([year0, year1])
        else:
            a.xaxis.set_ticks([])
        a.yaxis.set_ticks_position('left')
        a.tick_params(axis='y', pad=-3)
        a.set_xlabel('')
        a.text(.5, .99, delta, ha='center', va='top', transform=a.transAxes)

    # axs[0,0].set_ylabel('Storm surge exposure, people/year')
    # axs[0,0].set_xlabel('Forecast year')

    # f.text(.5, .99, 'Surge exposure trends:', ha='center', va='top')
    # f.text(.5, .95, '{} '.format(scenarios[0]), ha='right', va='center')
    # f.text(.5, .95, '{}'.format(scenarios[1]), ha='left', va='center')

    colors = list(iter(mpl.rcParams['axes.prop_cycle']))
    for s,c,x in zip(['Surge exposure trends:', scenarios[0], scenarios[1]], ['k', colors[0]['color'], colors[1]['color']], [.35, .53, .65]):
        f.text(x, .95, s, color=c)

    f.text(.035, .5, 'EV(annual surge exposure), people', fontsize=18, rotation=90, va='center', transform=f.transFigure)
    f.text(.5, .035, 'Outlook year', fontsize=18, ha='center', transform=f.transFigure)

    f.savefig(str(target[0]))
    plt.close(f)
    return 0


def plot_risk_quadrants(env, source, target):
    hazard = pandas.read_pickle(str(source[0])) # waves, discharge, surge?
    exposure = pandas.read_pickle(str(source[1])) # population (log?), or rslr
    vuln = pandas.read_pickle(str(source[2]))  # norm(norm(gdp) + norm(percap gdp))

    # hazard
    RCP = env['RCP']
    hazard_window = env['hazard_window']

    # vuln
    SSP = env['SSP']
    forecast = env['forecast']

    hazard = hazard.loc[(slice(None), RCP, hazard_window)]
    vuln = vuln.loc[:, (SSP, forecast)]

    hazard_name = env['hazard_name']
    exposure_name = env['exposure_name']
    vuln_name = env['vuln_name']
    delta_labels = env['delta_labels']

    df = pandas.DataFrame({exposure_name: exposure, hazard_name: hazard, vuln_name: vuln})
    x = df[exposure_name]
    y = df[hazard_name]
    s = df[vuln_name]

    mpl.style.use('ggplot')
    color = next(iter(mpl.rcParams['axes.prop_cycle']))['color']
    f, a = plt.subplots(1,1)

    marker_size = 400 * (s-s.min()) / (s.max()-s.min()) + 20
    alpha = .5
    lw = 1
    a.scatter(x=x, y=y, s=marker_size, facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
    a.scatter(x=x, y=y, s=marker_size, facecolor='none', edgecolor=color, lw=lw, zorder=2)
    dummy1, = a.plot([], [], marker='o', linestyle='none', markerfacecolor=color, markeredgecolor='none', alpha=alpha)
    dummy2, = a.plot([], [], marker='o', linestyle='none', markerfacecolor='none', markeredgecolor=color, lw=lw)

    lims = np.array(a.axis())
    adjw = .1 * (lims[1] - lims[0])
    adjh = .1 * (lims[3] - lims[2])
    newlims = lims + (-adjw, adjw, -adjh, adjh)
    a.axis(newlims)
    texts = []
    for d in delta_labels:
        a.scatter(x=x[d], y=y[d], s=marker_size[d], facecolor='none', edgecolor='.3', lw=lw*2, zorder=2)
        t = a.text(df.loc[d, exposure_name], df.loc[d, hazard_name], d, ha='center', va='center', alpha=.8)
        texts.append(t)
    if delta_labels:
        adjust_text(texts=texts, x=df.loc[:, exposure_name], y=df.loc[:, hazard_name], arrowprops=dict(arrowstyle='-', color='.3', shrinkA=0, shrinkB=0), ax=a, expand_text=(2.,2.), expand_points=(2.,2.), force_text=0.5, force_points=0.5)

    a.set_xlabel(exposure_name)
    a.set_ylabel(hazard_name)
    a.plot([0, newlims[1]], [0, 0], '0.3', lw=1, zorder=1)
    a.plot([0, 0], [0, newlims[3]], '0.3', lw=1, zorder=1)
    a.text(.95, .95, 'Marker size scales with {0}'.format(vuln_name.lower()), transform=a.transAxes, ha='right', va='center')

    f.savefig(str(target[0]))

    return 0


    # env.Command(
            # source=[config['hazards_index'],
                    # config['rsl_timeseries'],
                    # config['total_vuln']],
            # target=config['risk_quadrant_projection'],
            # action=pr.plot_risk_quad_projection,
            # deltas=common['deltas'],
            # plottype='lines',
            # )
def plot_risk_quad_projection(env, source, target):
    hazards = pandas.read_pickle(str(source[0]))
    rsl = pandas.read_pickle(str(source[1]))
    vuln = pandas.read_pickle(str(source[2]))

    deltas = env['deltas']
    plottype = env['plottype']
    plotdir = env['plotdir']
    RCPs = env['RCPs']
    SSPs = env['SSPs']
    hazard_name = env['hazard_name']
    exposure_name = env['exposure_name']
    vuln_name = env['vuln_name']
    date_labels = env['date_labels']
    rescale_hazard = env['rescale_hazard']

    color = next(iter(mpl.rcParams['axes.prop_cycle']))['color']

    datemapper = {
            'historical': 2000,
            'early21C': 2030,
            'late21C': 2080,
            }

    f, a = plt.subplots(1,1)

    # vnorm = mpl.colors.Normalize(vuln.loc[deltas,:].min().min(), vuln.loc[deltas,:].max().max())
    alpha = .5
    minh = 999
    maxh = -999
    minv = 999
    maxv = -999
    mine = 999
    maxe = -999
    # method = 'linecollection'
    method = 'fillbetween'
    if plotdir == 'hor':
        cols = ['e', 'h']
    elif plotdir == 'vert':
        cols = ['h', 'e']
    for d in deltas:
        for rcp in RCPs:
            for ssp in SSPs:
                h = hazards.loc[(d, ('none',)+rcp, slice(None))]
                e = rsl.loc[d, (slice(None), rcp)]
                v = vuln.loc[d, (ssp, slice(None))]
                if rescale_hazard:
                    h = h - h.iloc[0]
                h.index = h.index.droplevel('Delta').droplevel('RCP').rename('Year')
                e.index = e.index.droplevel('RCP')
                v.index = v.index.droplevel('SSP').rename('Year')
                h = h.rename_axis(datemapper)
                h = h.interpolate('index')
                h = h.fillna(method='backfill')
                h = h.fillna(method='pad')
                v = v.interpolate('index')
                v = v.fillna(method='backfill')
                v = v.fillna(method='pad')
                e = e.interpolate('index')
                e = e.fillna(method='backfill')
                e = e.fillna(method='pad')
                minh = min(minh, h.min())
                maxh = max(maxh, h.max())
                minv = min(minv, v.min())
                maxv = max(maxv, v.max())
                mine = min(mine, e.min())
                maxe = max(maxe, e.max())
    if plotdir == 'hor':
        buffh = (maxh - minh) * 0.1
        minh -= buffh
        maxh += buffh
        a.set_xlim(mine, maxe)
        a.set_ylim(minh, maxh)
    elif plotdir == 'vert':
        buffe = (maxe - mine) * 0.1
        mine -= buffe
        maxe += buffe
        a.set_xlim(minh, maxh)
        a.set_ylim(mine, maxe)
    vnorm = mpl.colors.Normalize(vmin=minv, vmax=maxv)
    for d in deltas:
        for rcp in RCPs:
            for ssp in SSPs:
                h = hazards.loc[(d, ('none',)+rcp, slice(None))]
                e = rsl.loc[d, (slice(None), rcp)]
                v = vuln.loc[d, (ssp, slice(None))]

                if rescale_hazard:
                    h = h - h.iloc[0]

                h.index = h.index.droplevel('Delta').droplevel('RCP').rename('Year')
                h = h.rename_axis(datemapper)
                e.index = e.index.droplevel('RCP')
                v.index = v.index.droplevel('SSP').rename('Year')

                df = pandas.DataFrame(dict(h=h, e=e, v=v))
                df = df.interpolate('index')
                df = df.fillna(method='backfill')
                df = df.fillna(method='pad')

                if np.isnan(df['v'].mean()):
                    continue

                if plottype == 'lines':
                    # import ipdb;ipdb.set_trace()
                    segments = map(lambda s: (s[0].tolist(), s[1].tolist()),
                            zip(df[cols].iloc[:-1,:].values,
                                df[cols].iloc[1:,:].values))
                    if method == 'linecollection':
                        lc = mpl.collections.LineCollection(
                                segments=segments,
                                linewidths=vnorm(df['v'])*20,
                                colors=color,
                                alpha=.5)
                        a.add_collection(lc)
                    if method == 'fillbetween':
                        x = []
                        y1 = []
                        y2 = []
                        widths = vnorm(df['v'])*(maxh-minh)/30
                        for seg, width in zip(segments, widths):
                            if seg[0] == seg[1]:
                                continue
                            coords0 = a.transData.transform(seg[0])
                            coords1 = a.transData.transform(seg[1])
                            theta = np.arctan2(coords1[1]-coords0[1], coords1[0]-coords0[0])
                            if plotdir == 'hor':
                                dy = (width/2.) / np.cos(theta)
                                x.append(seg[0][0])
                                y1.append(seg[0][1]-dy)
                                y2.append(seg[0][1]+dy)
                            elif plotdir == 'vert':
                                dy = (width/2.) / np.sin(theta)
                                x.append(seg[0][1])
                                y1.append(seg[0][0]-dy)
                                y2.append(seg[0][0]+dy)
                            print dy
                            if dy>.4:
                                import ipdb;ipdb.set_trace()
                        if plotdir == 'hor':
                            x.append(seg[1][0])
                            y1.append(seg[1][1]-dy)
                            y2.append(seg[1][1]+dy)
                            a.fill_between(x=x, y1=y1, y2=y2, alpha=.5)
                        elif plotdir == 'vert':
                            x.append(seg[1][1])
                            y1.append(seg[1][0]-dy)
                            y2.append(seg[1][0]+dy)
                            a.fill_betweenx(y=x, x1=y1, x2=y2, alpha=.5)

                    for i, ((date1, data1), (date2, data2)) in enumerate(itertools.izip(df.iloc[:-1,:].iterrows(), df.iloc[1:,:].iterrows())):
                        if method == 'lotsoflines':
                            lw = vnorm(data1['v']) * 5
                            a.plot([data1[cols[0]], data2[cols[0]]],
                                   [data1[cols[1]], data2[cols[1]]],
                                   lw=vnorm(data1['v'])*4,
                                   color=color,
                                   alpha=.5,
                                   solid_capstyle='butt')
                        if (date1 % date_labels == 0) and (date1 >= 2000):
                            a.text(data1['e'], data1['h'], date1, fontsize=8, ha='center', va='center')

                elif plottype == 'circles':
                    ms = vnorm(data1['v']) * 400 + 20
                    a.scatter(x=data1[cols[0]], y=data1[cols[1]], s=ms, facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
                    a.scatter(x=data1[cols[0]], y=data1[cols[1]], s=ms, facecolor='none', edgecolor=color, lw=1, zorder=2)
                else:
                    raise NotImplementedError

    if plottype == 'lines':
        a.text(.95, .95, 'Line width scales with {0}'.format(vuln_name.lower()), transform=a.transAxes, ha='right', va='center')
    elif plottype == 'circles':
        a.text(.95, .95, 'Marker size scales with {0}'.format(vuln_name.lower()), transform=a.transAxes, ha='right', va='center')
    if plotdir == 'hor':
        a.set_xlabel(exposure_name)
        a.set_ylabel(hazard_name)
    elif plotdir == 'vert':
        a.set_xlabel(hazard_name)
        a.set_ylabel(exposure_name)
    # a.plot([0, newlims[1]], [0, 0], '0.3', lw=1, zorder=1)
    # a.plot([0, 0], [0, newlims[3]], '0.3', lw=1, zorder=1)

    f.savefig(str(target[0]))

    return 0
