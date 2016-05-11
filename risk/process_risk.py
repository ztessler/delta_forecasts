import csv
import numpy as np
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
