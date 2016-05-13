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
    pops = exposure.loc[delta].unstack(level='Pop_Scenario').iloc[:, ::-1]
    pops.columns = pops.columns.rename('Population Growth Scenarios')
    pops = pops.rename(columns=lambda s: s.title())

    f, a = plt.subplots(1, 1, figsize=(12,8))
    pops.plot(ax=a, marker='o', markeredgecolor='none')
    a.set_ylabel('Storm surge exposure, people/year')
    a.set_xlabel('Forecast year')
    a.set_title('{}: Flood exposure trends due to population change'.format(delta))
    f.savefig(str(target[0]))
    plt.close(f)
    return 0
