import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas


def import_idi_vuln(env, target, source):
    def clean_name(s):
        return s.replace(' ','_').replace('-','_')

    with open(str(source[0]), 'r') as fd:
        fd.readline()
        fd.readline()
        csvfile = csv.DictReader(fd)

        deltas = pandas.read_pickle(str(source[1]))

        vuln = pandas.Series(index=deltas.index)
        for rec in csvfile:
            vuln[clean_name(rec['Delta'])] = float(rec['InvestmentDeficitIndex'])

    vuln.to_pickle(str(target[0]))
    return 0


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
    pops = pops.rename(columns=lambda s: s.title())

    color = next(iter(mpl.rcParams['axes.prop_cycle']))['color']

    f, a = plt.subplots(1, 1, figsize=(12,8))
    # pops.plot(ax=a, c=color, lw=1, legend=False)
    pops['Medium'].plot(ax=a, c=color, lw=3, legend=False)
    a.fill_between(pops.index, pops.iloc[:,0], pops.iloc[:,-1], color=color, alpha=.3)

    a.set_ylabel('Storm surge exposure, people/year')
    a.set_xlabel('Forecast year')
    a.set_title('{0}: Flood exposure trends ({1})'.format(delta, exp))
    f.savefig(str(target[0]))
    plt.close(f)
    return 0


def mk_plot_multiscenario_exposure(a, pops, scenarios, legend=True):
    mpl.style.use('ggplot')

    color_cycle = mpl.rcParams['axes.prop_cycle']

    dummy_fills = []
    for scenario, color in zip(scenarios, color_cycle):
        # pops[name].plot(ax=a, color=color['color'], lw=2, legend=False)
        pops[scenario, 'Medium'].plot(ax=a, color=color['color'], lw=3, legend=False)
        a.fill_between(pops.index, pops[scenario].iloc[:,0], pops[scenario].iloc[:,-1], color=color['color'], alpha=.3)
        if legend:
            dummy_fills.append(mpl.patches.Rectangle((0,0),1,1, fc=color['color'], ec='none'))
    if legend:
        a.legend(dummy_fills, scenarios, loc=2, title=pops.columns.names[0])

    return


def plot_surge_annual_exposure_multiscenario(env, target, source):
    exposures = [pandas.read_pickle(str(s)) for s in source]
    delta = env['delta']
    scenarios = env['scenarios']

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
    exposures = [pandas.read_pickle(str(s)) for s in source]

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
        return int(round(x, -int(np.floor(np.log10(abs(x))))))

    exposures = [pandas.read_pickle(str(s)) for s in source[:-1]]
    ranges = pandas.read_pickle(str(source[-1]))
    deltas = exposures[0].sum(level='Delta').dropna().index
    scenarios = env['scenarios']

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
        a.set_ylim(0, round_to_1digit(ranges.loc[delta, 'max']))
        a.yaxis.set_ticks([0, round_to_1digit(ranges.loc[delta, 'max'])])
        if i == len(deltas)-1:
            a.xaxis.set_ticks([year0, year1])
        else:
            a.xaxis.set_ticks([])
        a.yaxis.set_ticks_position('left')
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
