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


def surge_expected_expo(env, target, source):
    '''
    Compute expected flooding due to surge for any given year.   Adjusts to avoid double counting
    Estimated as:
        (prob of 25 year flood) * (exposure to 25 year flood) +
        (prob of 10 year flood - prob of 25 year flood) * (exposure to 10 year flood) +
        ...
    '''
    exposure = pandas.read_pickle(str(source[0]))

    def one_yr_expected(s):
        total_exp = 0.0
        prob_more_extreme = 0.0
        for ret_period, ret_level in s.sort_index(ascending=False).iteritems():
            if ret_level is np.nan:
                continue
            prob = 1./ret_period
            total_exp += (prob - prob_more_extreme) * ret_level
            prob_more_extreme += prob
        return total_exp

    exposure.apply(one_yr_expected, axis=0).to_pickle(str(target[0]))
    return 0
