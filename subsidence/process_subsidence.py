import pandas
import pint
import rasterio


def steady_state_subsidence(env, target, source):
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

    eustatic_slr = Q_(env['eustatic_slr'], 'mm/year')

    # 0 = rslr = slr + subsidence - sedimentation  # Ericson 2006 Eq. 1
    # subsidence = sedimentation - slr
    sedimentation = Qs * retention_frac / sed_density / area
    subsidence = sedimentation - eustatic_slr
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

    sub = 3 * drawdown/max_dd * natural_sub # Ericson 2006, sec. 4.2
    sub[sub<=0] = 0
    sub.to_pickle(str(target[0]))
    return 0
