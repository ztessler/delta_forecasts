import pandas
import pint


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


