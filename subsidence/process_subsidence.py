import csv
import numpy as np
import pandas
import geopandas
import fiona
import pint
import rasterio
import shapely.geometry as sgeom
import shapely.ops as sops
from interval import interval
from collections import defaultdict

from cStringIO import StringIO

def import_rslr_lit(env, target, source):
    '''
    Import rslr estimates from literture source.  Lit sources are given by a
    range (say, 5-20).  Calc mean of each range, estimate a standard deviation
    by weighting the range at the mean.  If "mean_weighting" is 2, the standard
    deviation of the range (3-9) is std([3,6,6,9])
    '''
    deltas = pandas.read_pickle(str(source[0]))
    mean_weighting = env['mean_weighting']

    data = pandas.DataFrame({'mean':[], 'std':[]})
    ranges = defaultdict(list)
    with open(str(source[1])) as f:
        reader = csv.DictReader(f)
        for entry in reader:
            est = entry['RSLR, mm/y']
            if '-' in est:
                est = map(float, est.split('-'))
                meanval = np.mean(est)
                for i in range(mean_weighting):
                    est.insert(1, meanval)
            else:
                try:
                    est = [float(est)]
                except ValueError:
                    continue
            ranges[entry['Delta']].extend(est)
    for delta, rslrs in ranges.iteritems():
        data.loc[delta, 'mean'] = np.nanmean(rslrs)
        if len(rslrs) > 1:
            data.loc[delta, 'std'] = np.nanstd(rslrs, ddof=1)
        else:
            data.loc[delta, 'std'] = np.nan
    # if only a single estimate, set standard deviation to the estimated rslr
    data.loc[data['std'].isnull(), 'std'] = data.loc[data['std'].isnull(), 'mean'] #data.mean()['std']
    data.to_pickle(str(target[0]))


def sed_aggradation(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    area = Q_(pandas.read_pickle(str(source[0])),
            'km**2')
    Qs = Q_(pandas.read_pickle(str(source[1]))
            .groupby(level='Delta')
            .sum(),
            'kg/s')

    # estimate from Blum and Roberts 2009, Nature GeoSci, mississippi value
    sed_density = Q_(1.5, 'g/cm**3') # g/cm**3, also in BT/km**3

    # estimate from same paper, range of 30-70%. references Ganges values of 30%, and 39-71%.
    # mississippi values from storage/load estimates are in this range. use 50%
    retention_frac = 0.5

    # biogenic sediment ??

    aggradation = Qs * retention_frac / sed_density / area
    aggradation.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def accomodation_space(env, target, source):
    deltas = pandas.read_pickle(str(source[0]))
    space = pandas.Series(index=deltas.index)
    shape_factor = env['shape_factor']
    with open(str(source[1])) as f:
        reader = csv.DictReader(f)
        for entry in reader:
            delta = entry['Delta']
            if delta in deltas:
                area = Q_(float(entry['Ad, km2']), 'km**2')
                depth = Q_(float(entry['Dsh, m']), 'm')
                space[delta] = (shape_factor * area * depth).to('km**3').magnitude
    # retention[retention.isnull()] = retention.mean()
    retention[retention.isnull()] = 1
    retention.to_pickle(str(target[0]))
    return 0


def import_sed_retention_ratio(env, target, source):
    deltas = pandas.read_pickle(str(source[0]))
    retention = pandas.Series(index=deltas.index)
    with open(str(source[1])) as f:
        reader = csv.DictReader(f)
        for entry in reader:
            delta = entry['Delta']
            if delta in deltas:
                retention[delta] = float(entry['TVs/OAdDsh'])
    # retention[retention.isnull()] = retention.mean()
    retention[retention.isnull()] = 1
    retention.to_pickle(str(target[0]))
    return 0


def steady_state_subsidence(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    aggradation = Q_(pandas.read_pickle(str(source[0])),
            'mm/year')

    # area = Q_(pandas.read_pickle(str(source[0])),
            # 'km**2')
    # Qs = Q_(pandas.read_pickle(str(source[1]))
            # .groupby(level='Delta')
            # .sum(),
            # 'kg/s')

    # # estimate from Blum and Roberts 2009, Nature GeoSci, mississippi value
    # sed_density = Q_(1.5, 'g/cm**3') # g/cm**3, also in BT/km**3

    # # estimate from same paper, range of 30-70%. references Ganges values of 30%, and 39-71%.
    # # mississippi values from storage/load estimates are in this range. use 50%
    # retention_frac = 0.5

    eustatic_slr = Q_(env['eustatic_slr'], 'mm/year')

    # 0 = rslr = slr + subsidence - sedimentation  # Ericson 2006 Eq. 1
    # subsidence = aggradation - slr
    # aggradation = Qs * retention_frac / sed_density / area
    subsidence = aggradation - eustatic_slr
    subsidence.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def clean_groundwater_stats(env, target, source):
    groundwater = pandas.read_pickle(str(source[0]))
    groundwater.fillna(0, inplace=True)
    groundwater.to_pickle(str(target[0]))
    return 0


def compute_drawdown(env, target, source):
    Q_ = pint.UnitRegistry().Quantity

    groundwater = Q_(pandas.read_pickle(str(source[0]))['mean'], 'm**3/year') * 1e6
    areas = Q_(pandas.read_pickle(str(source[1])), 'km**2')
    specific_yield = 0.2

    drawdown = groundwater / (areas * specific_yield)  # Ericson 2006 eq. 5
    drawdown.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def groundwater_subsidence(env, target, source):
    drawdown = pandas.read_pickle(str(source[0]))
    natural_sub = pandas.read_pickle(str(source[1]))

    max_dd = drawdown.max()
    if max_dd > 0:
        sub = 3.0 * drawdown/max_dd * natural_sub # Ericson 2006, sec. 4.2
        sub[sub<=0] = 0
    else:
        sub = 0.0 * natural_sub
    sub.to_pickle(str(target[0]))
    return 0


def oilgas_locations(env, target, source):
    oilgas_shpfile = str(source[0])
    deltas = geopandas.read_file(str(source[1])).set_index('Delta')

    with fiona.open(oilgas_shpfile, 'r') as oilgas_fields:
        fields = []
        for field in oilgas_fields:
            shape = sgeom.shape(field['geometry']).buffer(0)
            fields.append(shape)
    fields = sops.unary_union(fields)
    oilgas = deltas.intersects(fields)
    oilgas['Mississippi'] = True # set manually, dataset is only for outside the US
    oilgas.to_pickle(str(target[0]))
    return 0


def oilgas_subsidence(env, target, source):
    oilgas = pandas.read_pickle(str(source[0]))
    sub = oilgas * 1 #mm/year  Ericson 2006 sec 4.2
    sub.to_pickle(str(target[0]))
    return 0


def compute_rslr(env, target, source):
    aggradation = pandas.read_pickle(str(source[0]))
    natural_sub = pandas.read_pickle(str(source[1]))
    groundwater_sub = pandas.read_pickle(str(source[2]))
    oilgas_sub = pandas.read_pickle(str(source[3]))
    eustatic_slr = env['eustatic_slr']

    # Ericson 2006 Eq. 2
    rslr = eustatic_slr + natural_sub + groundwater_sub + oilgas_sub - aggradation

    eps = np.finfo(np.float).eps
    rslr[rslr<eps] = 0.0
    rslr.to_pickle(str(target[0]))


def rslr_regression_model(env, target, source):
    nat_sub = pandas.read_pickle(str(source[0]))
    agg = pandas.read_pickle(str(source[1]))
    drawdown = pandas.read_pickle(str(source[2]))
    oilgas = pandas.read_pickle(str(source[3]))
    rslr_lit = pandas.read_pickle(str(source[4]))
    eustatic_slr = env['eustatic_slr']

    # import statsmodels.formula.api as smf
    rslr_lit = rslr_lit.reindex(nat_sub.index).dropna()
    subsidence = rslr_lit - eustatic_slr

    drivers = pandas.DataFrame({'subsidence': subsidence,
                                'nat_sub': nat_sub,
                                'agg': agg,
                                'drawdown': drawdown,
                                'oilgas': oilgas.astype(float)})
    train = drivers.reindex(rslr_lit.index)

    nat_sub = nat_sub.reindex(rslr_lit.index)
    agg = agg.reindex(rslr_lit.index)
    drawdown = drawdown.reindex(rslr_lit.index)
    oilgas = oilgas.reindex(rslr_lit.index)

    model = smf.ols(formula='subsidence ~ nat_sub + agg + drawdown + oilgas', data=drivers)
    results = model.fit()
    print results.summary()
    # rslr_est = pandas.Series(index=deltas)

    from scipy.odr import Model, Data, ODR
    from scipy.stats import linregress
    rslr_lit = rslr_lit.reindex(nat_sub.index).dropna()
    subsidence = rslr_lit - eustatic_slr

    def func(params, x):
        # return np.dot(params, x)
        return params[0]*x[0,:] + params[1]*x[1,:] + params[2]*x[2,:] + params[3]*x[3,:] * params[4]*x[4,:]

    linear = Model(func)
    X = sm.add_constant(train.drop('subsidence', axis=1).values).T
    mydata = Data(X, subsidence.values)
    myodr = ODR(mydata, linear, beta0=np.ones(drivers.shape[1]))
    myresults = myodr.run()
    myresults.pprint()


