import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import pandas
import rasterio
import fiona
import pint


def rasterize_grand_dams(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    with rasterio.open(str(source[1]), 'r') as basins_rast:
        basins = basins_rast.read(1)
        meta = basins_rast.meta.copy()
    affine = meta['affine']
    resvol = np.zeros_like(basins)
    with fiona.open(str(source[0]), 'r') as dams:
        for dam in dams:
            props = dam['properties']
            if ((int(props['QUALITY'].split(':')[0]) <= 3) and #verified, good, or fair
                    (props['CAP_MCM'] > 0)):
                lon = props['LONG_DD']
                lat = props['LAT_DD']
                vol = Q_(1e6 * props['CAP_MCM'], 'm**3').to('km**3').magnitude
                x, y = map(np.floor, ~affine * (lon, lat))
                resvol[y, x] += vol

    with rasterio.open(str(target[0]), 'w', **meta) as resvol_rast:
        resvol_rast.write(resvol, 1)
    return 0


def compute_I(env, target, source):
    Ag = pandas.read_pickle(str(source[0]))
    I = (1 + 0.09*Ag)
    I.to_pickle(str(target[0]))
    return 0


def res_trapping(env, target, source):
    utilization = 0.67
    with rasterio.open(str(source[0]), 'r') as resrast,\
             rasterio.open(str(source[1]), 'r') as disrast,\
             rasterio.open(str(source[2]), 'r') as basinrast:
        kwargs = resrast.meta.copy()
        resvol = resrast.read(1, masked=True) * utilization * (1000**3) # convert km**3 to m**3
        dis = disrast.read(1, masked=True)  # m**3 / s
        basins = basinrast.read(1, masked=True)
    resvol.mask[resvol==0] = True
    basin_ids = pandas.read_pickle(str(source[3]))

    Te = []
    for delta, basin_id in basin_ids.index:
        pix = np.logical_and(basins == basin_id, resvol>0)
        res_time = resvol[pix].sum() / dis[basins==basin_id].max() / (60*60*24*365)
        Te.append(1 - (0.05/np.sqrt(res_time)))
    Te = [te if (te is not np.ma.masked and te >= 0) else 0 for te in Te]

    Te = pandas.Series(Te, index=basin_ids.index)
    Te.to_pickle(str(target[0]))
    return 0


                # env.Command(
                        # source=[config['reservoir_adj_source'][1],
                                # config['delta_zeros']],
                        # target=config['reservoir_adj'].format(ver='', ext='pd'),
                        # action=ps.parse_zarfl_xls)
def parse_zarfl_xls(env, target, source):
    zarfl = pandas.read_excel(str(source[0]), sheetname='Table S7').iloc[:, 1:]
    deltas = pandas.read_pickle(str(source[1]))

    zarfl.columns = np.r_[zarfl.iloc[0,0], zarfl.iloc[1,1:]]
    zarfl = zarfl.set_index('Major river basin')
    zarfl = zarfl.iloc[2:, :]

    cols =  map(lambda s: s.replace(' ', '')
                          .replace(',', '')
                          .replace(u'\u2013', '-'),
                zarfl.columns)
    zarfl.columns = pandas.CategoricalIndex(cols, categories=cols, ordered=True)
    changename = {
            u'Hong (Red River)': 'Hong',
            u'Ganges-Brahmaputra': 'Ganges',
            u'Tigris \u2013 Euphrates': 'Shatt_el_Arab',
            }
    zarfl.rename(index=lambda d: changename.get(d, d), inplace=True)
    zarfl = zarfl.reindex(index=deltas.index).dropna(axis=0, how='all')

    def split_dam_counts(s):
        if '(' not in str(s):
            return [int(s), 0]
        else:
            return map(int, s.replace(')','').split('('))
    zarfl_new = zarfl.applymap(split_dam_counts).applymap(lambda t: t[0])
    zarfl_old = zarfl.applymap(split_dam_counts).applymap(lambda t: t[1])

    multicols = pandas.MultiIndex.from_product([zarfl.columns, ['new', 'old']], names=['Discharge class', 'Scenario'])
    zarfl = pandas.DataFrame(0, index=zarfl.index, columns=multicols)
    zarfl.loc[:, (slice(None), 'new')] = zarfl_new.values
    zarfl.loc[:, (slice(None), 'old')] = zarfl_old.values
    zarfl.to_pickle(str(target[0]))
    return 0


def add_new_reservoirs(env, target, source):
    '''
    Place new dams in along river network.  Finds location on river network
    with discharge closest to mean of the zarfl discharge class.  Dam location
    not considered, just underlying discharge - OK given calculation of res_trapping,
    which computes discharge-weighted average of reservoir trapping efficiency
    '''
    with rasterio.open(str(source[0])) as ressrc:
        res_meta = ressrc.meta
        res = ressrc.read(1)

    res_adj = pandas.read_pickle(str(source[1]))

    with rasterio.open(str(source[2])) as dissrc:
        dis = dissrc.read(1)

    with rasterio.open(str(source[3])) as basinsrc:
        basins = basinsrc.read(1)

    basinids = pandas.read_pickle(str(source[4]))

    def dis_class_endmembers(s):
        return map(float, s.replace('>','').split('-'))

    def dis_class_mean(s):
        return np.mean(dis_class_endmembers(s))

    # estimate typical dam sizes based on river discharge
    # assume new dams will be similar sizes with respect to riv dis as exisiting dams
    new_vols = pandas.Series(0.0, index=res_adj.columns.levels[res_adj.columns.names.index('Discharge class')])
    for dis_class in new_vols.index:
        endmembers = dis_class_endmembers(dis_class)
        if len(endmembers) == 2:
            dam_locs = np.logical_and(res>0, np.logical_and(dis>=endmembers[0], dis<endmembers[1]))
        elif len(endmembers) == 1:
            dam_locs = np.logical_and(res>0, dis>endmembers)
        else:
            raise ValueError
        new_vols[dis_class] = res[dam_locs].mean()

    for delta, deltabasins in basinids.groupby(level='Delta'):
        if delta not in res_adj.index:
            continue
        mask = np.ones_like(res)
        for _, basinid in deltabasins.index:
            mask = np.logical_and(mask, basins!=basinid)
        mask[res>0] = True   # mark existing reservoirs
        # loop over discharge classes in reverse order so biggest dams get placed first. not necessary unless very few valid locs
        for (dis_class, res_in_class) in reversed(list(res_adj.groupby(axis=1, level='Discharge class', sort=True))):
            num_new_res = res_in_class.loc[delta, (dis_class, 'new')]
            for i in range(num_new_res):
                valid_dis = np.ma.masked_where(mask, dis)
                new_loc = np.unravel_index(np.argmin(np.abs(valid_dis - dis_class_mean(dis_class))), dis.shape)
                res[new_loc] = new_vols.loc[dis_class]
                mask[new_loc] = True

    with rasterio.open(str(target[0]), 'w', **res_meta) as resout:
        resout.write(res, 1)
    return 0


def compute_res_potential(env, source, target):
    with rasterio.open(str(source[0]), 'r') as rast:
        basins = rast.read(1)
    with rasterio.open(str(source[1]), 'r') as rast:
        relief = rast.read(1)
    with rasterio.open(str(source[2]), 'r') as rast:
        runoff = rast.read(1)
    basinids = pandas.read_pickle(str(source[3]))

    relief[relief < 0] = 0
    runoff[runoff < 0] = 0
    potential = relief * runoff

    basin_potential = pandas.Series(index=basinids.index)
    for (delta, basinid) in basinids.index:
        basin_potential[(delta, basinid)] = potential[basins==basinid].sum()

    basin_potential.to_pickle(str(target[0]))
    return 0


# def compute_basin_res_utilization(env, source, target):
    # with rasterio.open(str(source[0]), 'r') as rast:
        # res = rast.read(1)
    # with rasterio.open(str(source[1]), 'r') as rast:
        # basins = rast.read(1)
    # basinids = pandas.read_pickle(str(source[2]))
    # potential = pandas.read_pickle(str(source[3]))

    # basin_resvol = pandas.Series(index=basinids.index)
    # for (delta, basinid) in basinids.index:
        # basin_resvol[(delta, basinid)] = res[basins==basinid].sum()

    # utilization = basin_resvol / potential
    # utilization.to_pickle(str(target[0]))
    # return 0


def scale_reservoirs_by_utilization(env, source, target):
    with rasterio.open(str(source[0]), 'r') as rast:
        res = rast.read(1)
        meta = rast.meta
        meta['transform'] = meta['affine']
    with rasterio.open(str(source[1]), 'r') as rast:
        basins = rast.read(1)
    potential = pandas.read_pickle(str(source[2]))
    ref_basin = env['ref_basin']
    ref_basinid = potential.loc[ref_basin].index.sort_values(ascending=True)[0]

    basin_resvol = pandas.Series(index=potential.index)
    for (delta, basinid) in potential.index:
        basin_resvol[(delta, basinid)] = res[basins==basinid].sum()
    utilization = basin_resvol / potential
    ref_utilization = utilization[(ref_basin, ref_basinid)]

    res_adj = np.zeros_like(res)
    for (delta, basinid) in potential.index:
        basin_utilization = utilization[(delta, basinid)]
        if basin_utilization > 0:
            scaling = ref_utilization / basin_utilization
            res_adj[basins==basinid] = res[basins==basinid] * scaling
        else:
            # no reservoirs in this basin. add a single one of necessary size to some point in basin
            # other points already zero
            y, x = zip(*np.where(basins==basinid))[0]
            res_adj[y, x] = ref_utilization * potential[(delta, basinid)]

    utilization.to_pickle(str(target[0]))
    with rasterio.open(str(target[1]), 'w', **meta) as fout:
        fout.write(res_adj, 1)
    return 0


def compute_Eh(env, target, source):
    gdp = pandas.read_pickle(str(source[0]))
    popdens = pandas.read_pickle(str(source[1]))
    Eh = pandas.Series(1.0, index=gdp.index)
    Eh[np.logical_and(gdp>15000, popdens>200)] = 0.3
    Eh[np.logical_and(gdp<15000, popdens>200)] = 2
    Eh.to_pickle(str(target[0]))
    return 0


def compute_B(env, target, source):
    I = pandas.read_pickle(str(source[0]))
    L = pandas.read_pickle(str(source[1]))
    Te = pandas.read_pickle(str(source[2]))
    Eh = pandas.read_pickle(str(source[3]))
    B = I * L * (1 - Te) * Eh
    B.to_pickle(str(target[0]))
    return 0


def compute_Qs(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    B = pandas.read_pickle(str(source[0]))
    Q = Q_(pandas.read_pickle(str(source[1])), 'm**3/s').to('km**3/year').magnitude
    A = pandas.read_pickle(str(source[2]))
    R = pandas.read_pickle(str(source[3]))
    T = pandas.read_pickle(str(source[4]))
    w = 0.02 # for Qs units of kg/s.  Use w=0.0006 for MT/yt

    T[T<2] = 2
    Qs = w * B * Q**0.31 * A**0.5 * R * T
    Qs.to_pickle(str(target[0]))
    return 0


def plot_delta_scalars(env, target, source):
    mpl.style.use('ggplot')
    scenarios = env.get('scenarios', None)
    if not isinstance(scenarios, list):
        scenarios = [scenarios]
    ylabel = env['ylabel']
    xlabel = env['xlabel']
    title = env['title']
    logy = env.get('logy', False)
    exp_num = env.get('exp_num', None) # for color cycle control
    nsources = env['nsources']

    qs0 = pandas.read_pickle(str(source[0])).groupby(level='Delta').sum()
    if nsources > 1:
        qs1 = pandas.read_pickle(str(source[1])).groupby(level='Delta').sum()
        df = pandas.DataFrame({scenarios[0]:qs0, scenarios[1]:qs1},
                columns=[scenarios[0], scenarios[1]])
    else:
        df = pandas.DataFrame({scenarios[0]:qs0},
                columns=[scenarios[0]])
    df = df.sort_values(by=scenarios[0], ascending=False)
    df = df.drop('Congo')

    f, a = plt.subplots(1, 1, figsize=(16,8))
    if exp_num is None:
        df.plot(kind='bar', logy=logy, ax=a)
    else:
        color = next(itertools.islice(iter(mpl.rcParams['axes.prop_cycle']), exp_num, exp_num+1))['color']
        df.plot(kind='bar', logy=logy, ax=a, color=color)

    a.set_ylabel(ylabel)
    a.set_xlabel(xlabel)
    a.set_title(title)
    plt.tight_layout()
    f.savefig(str(target[0]))
    plt.close(f)

    return 0


def plot_scalars_percent_change(env, target, source):
    mpl.style.use('ggplot')
    scenarios = env['scenarios']
    ylabel = env['ylabel']
    xlabel = env['xlabel']
    title = env['title']

    qs0 = pandas.read_pickle(str(source[0])).groupby(level='Delta').sum()
    qs1 = pandas.read_pickle(str(source[1])).groupby(level='Delta').sum()

    change = (qs1-qs0)/qs0 * 100.
    change = change.drop('Congo')
    if change.mean() > 0:
        ascending = False
    else:
        ascending = True
    change = change.sort_values(ascending=ascending)
    f, a = plt.subplots(1, 1, figsize=(16,8))
    change.plot(kind='bar', ax=a)
    a.set_ylabel(ylabel)
    a.set_xlabel(xlabel)
    a.set_title(title + ', {} to {}'.format(scenarios[0], scenarios[1]))
    plt.tight_layout()
    f.savefig(str(target[0]))
    plt.close(f)

    return 0


def plot_rslr_timeseries(env, source, target):
    mpl.style.use('ggplot')
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    scenarios = env['scenarios']
    deltas = env['deltas']
    slr_cur = env['slr_cur'] # 'mm/year'
    slr_2100_RCP2p6 = env['slr_2100_RCP2p6'] # 'mm/year'
    slr_2100_RCP8p5 = env['slr_2100_RCP8p5'] # 'mm/year'

    years = env['years']
    years = pandas.Series(range(years[0], years[1]+1))
    yr_index = pandas.Series(years.index) # 'year'

    rslr_land_A = pandas.read_pickle(str(source[0])) # 'mm/year'
    rslr_land_B = pandas.read_pickle(str(source[1])) # 'mm/year'

    slr_const = pandas.Series(slr_cur, index=yr_index) # 'mm/year'
    sl_const = (slr_const.cumsum() - slr_const[0]) / 1000. # 'm'

    slr_2p6 = pandas.Series(np.linspace(slr_cur, slr_2100_RCP2p6, len(yr_index)), index=yr_index) # 'mm/year'
    sl_2p6 = (slr_2p6.cumsum() - slr_2p6[0]) / 1000. # 'm'

    slr_8p5 = pandas.Series(np.linspace(slr_cur, slr_2100_RCP8p5, len(yr_index)), index=yr_index) # 'mm/year'
    sl_8p5 = (slr_8p5.cumsum() - slr_8p5[0]) / 1000. # 'm'

    fig, axs = plt.subplots(len(deltas), 1, figsize=(4, 3*len(deltas)))
    mpl.rcParams.update({
                         'axes.labelsize': 'small',
                         'axes.titlesize': 'medium',
                         'legend.fontsize': 'small'})
    linestyle = {'const': '-', '2p6': '--', '8p5': ':'}
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i, (delta, ax) in enumerate(zip(deltas, axs)):
        for j, (rslr, c) in enumerate(zip([rslr_land_A, rslr_land_B], mpl.rcParams['axes.prop_cycle'])):
            rslr_const = (yr_index * (rslr[delta] + slr_cur)) / 1000. # 'm'
            rslr_2p6 = ((yr_index * rslr[delta]) / 1000.) + sl_2p6 # 'm'
            rslr_8p5 = ((yr_index * rslr[delta]) / 1000.) + sl_8p5 # 'm'
            rslr_const.index = years
            rslr_2p6.index = years
            rslr_8p5.index = years

            ax.plot(rslr_const, color=c['color'], lw=2, linestyle=linestyle['const'], label=scenarios[j])
            ax.plot(rslr_const, color=c['color'], lw=2, linestyle=linestyle['const'])
            ax.plot(rslr_2p6, color=c['color'], lw=2, linestyle=linestyle['2p6'])
            ax.plot(rslr_8p5, color=c['color'], lw=2, linestyle=linestyle['8p5'])

            ax.set_title(delta)
            ax.set_ylabel('Relative sea-level rise, (m)', fontsize='small')

        ax.plot([], [], color='.5', lw=2, linestyle=linestyle['const'], label='Const SLR')
        ax.plot([], [], color='.5', lw=2, linestyle=linestyle['2p6'], label='RCP 2.6 SLR')
        ax.plot([], [], color='.5', lw=2, linestyle=linestyle['8p5'], label='RCP 8.5 SLR')
        if i == 0:
            ax.legend(loc=2, frameon=False, handlelength=3)
        if i == len(deltas)-1:
            ax.set_xlabel('Year', fontsize='small')
        else:
            ax.xaxis.set_ticklabels([])
        ax.tick_params(axis='both', which='major', labelsize='small')

    fig.tight_layout()
    fig.savefig(str(target[0]))
    return 0
