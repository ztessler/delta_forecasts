import numpy as np
import pandas
import rasterio
import pint


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
    residence_time = resvol / dis / (60 * 60 * 24 * 365) # years
    trapping_eff = 1 - 0.05/np.sqrt(residence_time)

    Te = []
    for delta, basin_id in basin_ids.index:
        pix = np.logical_and(basins == basin_id, ~trapping_eff.mask)
        Te.append((trapping_eff[pix] * dis[pix]).sum() / dis[pix].sum())
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

    # assign columns as mean of given ranges
    zarfl.columns = map(lambda l: np.mean(map(float, l)),
                        map(lambda s: s.replace(' ', '')
                                       .replace(',', '')
                                       .replace('>', '')
                                       .split(u'\u2013'),
                            zarfl.columns))
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

    multicols = pandas.MultiIndex.from_product([zarfl.columns, ['new', 'old']])
    zarfl = pandas.DataFrame(0, index=zarfl.index, columns=multicols)
    zarfl.loc[:, (slice(None), 'new')] = zarfl_new.values
    zarfl.loc[:, (slice(None), 'old')] = zarfl_old.values
    zarfl.to_pickle(str(target[0]))
    return 0


def add_new_reservoirs(env, target, source):
    pass


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
