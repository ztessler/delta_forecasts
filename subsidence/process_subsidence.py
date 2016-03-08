import numpy as np
import pandas
import geopandas
import fiona
import pint
import rasterio
import shapely.geometry as sgeom
import shapely.ops as sops


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

    aggradation = Qs * retention_frac / sed_density / area
    aggradation.to('mm/year').magnitude.to_pickle(str(target[0]))
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

