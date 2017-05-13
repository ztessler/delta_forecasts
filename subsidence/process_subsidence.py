import sys
import csv
import numpy as np
import pandas
import geopandas
import fiona
import pint
import rasterio
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, RESAMPLING
from rasterstats import zonal_stats
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import shapely.ops as sops
from affine import Affine
from interval import interval
from collections import defaultdict

from scipy import odr
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

    data = pandas.DataFrame({'mean':[], 'std':[], 'min':[], 'max':[]}, dtype='float')
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
        data.loc[delta, 'min'] = np.min(rslrs)
        data.loc[delta, 'max'] = np.max(rslrs)
    # if only a single estimate, set standard deviation to the estimated rslr
    data.loc[data['std'].isnull(), 'std'] = data.loc[data['std'].isnull(), 'mean'] #data.mean()['std']
    data.to_pickle(str(target[0]))


def import_accomodation_space(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    deltas = pandas.read_pickle(str(source[0]))
    space = pandas.Series(index=deltas.index)
    shape_factor = 0.5 #env['shape_factor']
    with open(str(source[1])) as f:
        reader = csv.DictReader(f)
        for entry in reader:
            delta = entry['Delta']
            if delta in deltas:
                area = Q_(float(entry['Ad, km2']), 'km**2')
                depth = Q_(float(entry['Dsh, m']), 'm')
                space[delta] = (shape_factor * area * depth).to('km**3').magnitude
    space[space.isnull()] = space.mean()
    space.to_pickle(str(target[0]))
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
    # sed_density = Q_(1.5, 'g/cm**3') # g/cm**3, also in BT/km**3
    sed_density = Q_(env['sed_dens'], 'g/cm**3')
    sed_porosity = env['sed_poro']

    # estimate from same paper, range of 30-70%. references Ganges values of 30%, and 39-71%.
    # mississippi values from storage/load estimates are in this range. use 50%
    # retention_frac = 0.5
    retention_frac = env['retention']

    # biogenic sediment ??

    aggradation = ((Qs * retention_frac / sed_density) / (1.0 - sed_porosity)) / area
    aggradation.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def sed_aggradation_variable_retention(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    area = Q_(pandas.read_pickle(str(source[0])),
            'km**2')
    Qs = Q_(pandas.read_pickle(str(source[1]))
            .groupby(level='Delta')
            .sum(),
            'kg/s')
    retention_frac = pandas.read_pickle(str(source[2]))

    # estimate from Blum and Roberts 2009, Nature GeoSci, mississippi value
    sed_density = Q_(1.5, 'g/cm**3') # g/cm**3, also in BT/km**3
    sed_porosity = env['sed_poro']

    # biogenic sediment ??

    aggradation = ((Qs * retention_frac / sed_density) / (1.0 - sed_porosity)) / area
    aggradation.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def sed_aggradation_with_progradation(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    area = Q_(pandas.read_pickle(str(source[0])),
            'km**2')
    Qs = Q_(pandas.read_pickle(str(source[1]))
            .groupby(level='Delta')
            .sum(),
            'kg/s')

    sed_density = Q_(env['sed_dens'], 'g/cm**3')
    sed_porosity = env['sed_poro']
    retention_frac = env['retention']
    delta_age = Q_(env['delta_age'], 'years')

    growth_rate = area / delta_age
    new_area = area + (growth_rate * Q_(1.0, 'year'))

    aggradation = ((Qs * retention_frac / sed_density) / (1.0 - sed_porosity)) / new_area
    aggradation.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def steady_state_subsidence(env, target, source):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    aggradation = Q_(pandas.read_pickle(str(source[0])),
            'mm/year')
    eustatic_slr = Q_(env['eustatic_slr'], 'mm/year')

    # 0 = rslr = slr + subsidence - sedimentation  # Ericson 2006 Eq. 1
    # subsidence = aggradation - slr
    subsidence = aggradation - eustatic_slr
    subsidence.to('mm/year').magnitude.to_pickle(str(target[0]))
    return 0


def const_nat_subsidence(env, target, source):
    ones = pandas.read_pickle(str(source[0]))
    rate = env['sub_rate'] * ones
    rate.to_pickle(str(target[0]))
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


def extract_gia(env, source, target):
    deltas = geopandas.read_file(str(source[0])).set_index('Delta')
    with rasterio.open(str(source[1]), 'r') as rast:
        gia_rast = rast.read(1)
        raw_crs = rast.crs
        raw_affine = rast.affine
        raw_bounds = rast.bounds
        raw_width = rast.width
        raw_height = rast.height
        raw_nodata = rast.nodata
    gia_rast = np.roll(gia_rast, 180)
    raw_affine = Affine.translation(-180, 0) * raw_affine

    gia = pandas.Series(0.0, index=deltas.index)
    for delta in deltas.index:
        lon, lat = np.array(deltas.centroid[delta])
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=lon, central_latitude=lat)
        delta_proj = deltas.loc[[delta]].to_crs(proj.proj4_params).convex_hull
        bounds_proj = delta_proj.buffer(50000).bounds.squeeze()
        bounds = transform_bounds(proj.proj4_params, raw_crs, *bounds_proj)
        lon1, lat1, lon2, lat2 = bounds
        x1, y1 = ~raw_affine * (lon1, lat1)
        x1 = int(np.floor(x1))
        y1 = int(np.ceil(y1))
        x2, y2 = ~raw_affine * (lon2, lat2)
        x2 = int(np.ceil(x2))
        y2 = int(np.floor(y2))

        dst_affine, dst_width, dst_height = calculate_default_transform(
                raw_crs, proj.proj4_params, x2-x1, y1-y2,
                *bounds)
        gia_proj = np.ones((dst_height, dst_width), dtype=rasterio.float64)
        reproject(gia_rast, gia_proj, raw_affine, raw_crs, raw_nodata,
            dst_affine, proj.proj4_params, raw_nodata, RESAMPLING.bilinear)
        if len(gia_proj.flatten()) > 1:
            stats = zonal_stats(delta_proj, gia_proj, affine=dst_affine, nodata=raw_nodata, all_touched=True)
            meanval = stats[0]['mean']
        else:
            meanval = gia_proj[0,0]

        gia[delta] = meanval

    gia.to_pickle(str(target[0]))
    return 0


def compute_rslr(env, target, source):
    aggradation = pandas.read_pickle(str(source[0]))
    natural_sub = pandas.read_pickle(str(source[1]))
    groundwater_sub = pandas.read_pickle(str(source[2]))
    oilgas_sub = pandas.read_pickle(str(source[3]))
    gia_uplift = pandas.read_pickle(str(source[4]))
    eustatic_slr = env['eustatic_slr']

    # Ericson 2006 Eq. 2
    rslr = eustatic_slr + natural_sub + groundwater_sub + oilgas_sub - gia_uplift - aggradation

    eps = np.finfo(np.float).eps
    rslr.to_pickle(str(target[0]))


def retention_merge_to_df(env, source, target):
    low = pandas.read_pickle(str(source[0]))
    mid = pandas.read_pickle(str(source[1]))
    high = pandas.read_pickle(str(source[2]))

    ci = pandas.CategoricalIndex(['low', 'mid', 'high'], ordered=True)
    merged = pandas.DataFrame({'low': low, 'mid': mid, 'high': high},
                              columns=ci)
    merged.to_pickle(str(target[0]))
    return 0


def compute_retention_from_rslr_lit(env, source, target):
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    rslr_lit = Q_(pandas.read_pickle(str(source[0])), 'mm/year')
    Qs_prist = Q_(pandas.read_pickle(str(source[1])).groupby(level='Delta').sum(), 'kg/s')
    Qs_contemp = Q_(pandas.read_pickle(str(source[2])).groupby(level='Delta').sum(), 'kg/s')
    groundwater_sub = Q_(pandas.read_pickle(str(source[3])), 'mm/year')
    oilgas_sub = Q_(pandas.read_pickle(str(source[4])), 'mm/year')
    gia_uplift = Q_(pandas.read_pickle(str(source[5])), 'mm/year')
    area = Q_(pandas.read_pickle(str(source[6])), 'km**2')
    dens = Q_(env['sed_dens'], 'g/cm**3')
    sed_porosity = env['sed_poro']
    prist_slr = Q_(env['prist_slr'], 'mm/year')
    contemp_slr = Q_(env['contemp_slr'], 'mm/year')

    # RSLR_contemp = slr_contemp + subsidence_prist + sub_groundwater + sub_oilgas - gia - agg_contemp
    # RSLR_contemp = slr_contemp + (agg_prist - slr_prist) + sub_groundwater + sub_oilgas - gia - agg_contemp
    # RSLR_contemp = slr_contemp + ((Qs_prist * retention / dens / area) - slr_prist) + sub_groundwater + sub_oilgas - gia - (Qs_contemp * retention / dens / area)
    # RSLR_contemp = (slr_contemp - slr_prist + sub_groundwater + sub_oilgas - gia) + ((Qs_prist - Qs_contemp) * retention / dens / area)
    # (Qs_contemp - Qs_prist) * (retention / dens / area) = slr_contemp - slr_prist + sub_groundwater + sub_oilgas - gia - RSLR_contemp
    # retention = (slr_contemp - slr_prist + sub_groundwater + sub_oilgas - gia - RSLR_contemp) / (Qs_contemp - Qs_prist) * dens * area
    retention = ((contemp_slr - prist_slr + groundwater_sub + oilgas_sub - gia_uplift - rslr_lit['mean']) / (((Qs_contemp - Qs_prist) * dens)*(1.0-sed_porosity)) * area).to('').magnitude
    retention.to_pickle(str(target[0]))
    return 0


def rslr_regression_model(env, target, source):
    nat_sub = pandas.read_pickle(str(source[0]))
    agg = pandas.read_pickle(str(source[1]))
    drawdown_sub = pandas.read_pickle(str(source[2]))
    oilgas_sub = pandas.read_pickle(str(source[3]))
    rslr_lit = pandas.read_pickle(str(source[4]))
    eustatic_slr = env['eustatic_slr']

    deltas = nat_sub.index

    rslr_lit = rslr_lit.reindex(deltas).dropna()
    subsidence = rslr_lit['mean'] - eustatic_slr

    accel_sub = drawdown_sub + oilgas_sub

    drivers = pandas.DataFrame({'subsidence': subsidence,
                                'nat_sub': nat_sub,
                                'agg': agg,
                                # 'drawdown': drawdown,
                                # 'oilgas': oilgas.astype(float)})
                                'accel_sub': accel_sub})
    cols = ['nat_sub', 'agg', 'accel_sub']#, 'drawdown', 'oilgas']
    train = drivers.reindex(rslr_lit.index)

    # remove any data columns with all constant values
    var0 = train.var(0)
    for col in list(cols):
        if var0[col] == 0:
            train = train.drop(col, axis=1)
            cols.remove(col)

    #http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    class Capturing(list):
	def __enter__(self):
	    self._stdout = sys.stdout
	    sys.stdout = self._stringio = StringIO()
	    return self
	def __exit__(self, *args):
	    self.extend(self._stringio.getvalue().splitlines())
	    sys.stdout = self._stdout

    model = smf.ols(formula='subsidence ~ nat_sub + agg + accel_sub', data=train)
    ols_results = model.fit()
    with Capturing() as output:
        print ols_results.summary()


    def linfunc(params, x):
        return np.dot(params, x)

    linmodel = odr.Model(linfunc)
    X = sm.add_constant(train[cols].values).T
    mydata = odr.RealData(x=X, y=train['subsidence'].values, sy=rslr_lit['std'].values)
    myodr = odr.ODR(mydata, linmodel, beta0=ols_results.params[['Intercept']+cols])
    myodr.set_job(fit_type=0) # 0 for ODR, 2 for OLS
    odr_results = myodr.run()

    with Capturing(output) as output:
        odr_results.pprint()

    rslr_modeled = pandas.Series(index=deltas)
    for delta in deltas:
        rslr_modeled[delta] = linfunc(odr_results.beta, np.r_[1, drivers[cols].loc[delta]])
    rslr_modeled.to_pickle(str(target[0]))

    with Capturing(output) as output:
        for delta in train.index:
            X = sm.add_constant(train.drop(delta)[cols].values).T
            subdata = odr.RealData(X, train['subsidence'].drop(delta).values, sy=rslr_lit['std'].drop(delta).values)
            subodr = odr.ODR(subdata, linmodel, beta0=ols_results.params[['Intercept']+cols])
            subodr.set_job(fit_type=0)
            subresults = subodr.run()

            odrpred = linfunc(odr_results.beta, np.r_[1, train.loc[delta, cols]])
            subodrpred = linfunc(subresults.beta, np.r_[1, train.loc[delta, cols]])
            print '{}:'.format(delta)
            print '\t{}'.format(str(train.loc[delta,cols]).replace('\n','\n\t'))
            print '\ttarget:\t\t\t{}'.format(rslr_lit.loc[delta, 'mean'])
            print '\tmodeled full:\t\t{}'.format(odrpred)
            print '\tmodeled test:\t\t{}'.format(subodrpred)
            print '\tparams test:\t\t{}'.format(subresults.beta)

    with open(str(target[1]), 'w') as fout:
        fout.write('\n'.join(output))

    return 0

