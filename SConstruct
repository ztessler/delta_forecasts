# vim: set fileencoding=UTF-8 :
# vim:filetype=python

import os
import csv

SetOption('max_drift', 1)

env = Environment(ENV=os.environ)
env.Decider('MD5-timestamp')
Export('env')

def clean_delta_name(delta):
    return delta.replace(' ','_').replace('-','_')

with open('deltaIDs.csv', 'r') as deltaIDs:
    next(deltaIDs)
    next(deltaIDs)
    deltas = {}
    reader = csv.DictReader(deltaIDs)
    for d in reader:
        deltas[clean_delta_name(d['Delta'])] = int(d['deltaID'])

deltas = { # for testing
        'Mekong': 26,
        'Nile': 30,
         }
Export('deltas')

years = [2000] #, 2005, 2010, 2015, 2020]
Export('years')


# EXPERIMENT configs
experiments = {
        'default': {
            'reservoirs': '#data/Global/reservoir.tif',
            'dis_raster': '#data/Global/discharge.tif',
            'discharge': '#data/Global/upstream_discharge.pd',
            'Te': '#data/Global/upstream_trapping_eff.pd',
            'gdp': '#data/Global/upstream_per_capita_gdp.pd',
            'pop': '#data/Global/upstream_pop_dens_2000.pd',
            'Eh': '#data/Global/bqart_Eh.pd',
            'B': '#data/Global/bqart_B.pd',
            'Qs': '#data/Global/bqart_Qs.pd',
            },
        'contemp': {
            'Qs': '#data/Global/experiments/contemp/bqart_Qs.pd',
            },
        'pristine': {
            'Te': '#data/Global/upstream_zeros.pd',
            'Eh': '#data/Global/upstream_ones.pd',
            'B': '#data/Global/experiments/pristine/bqart_B.pd',
            'Qs': '#data/Global/experiments/pristine/bqart_Qs.pd',
            }
        }
for experiment in experiments.keys():
    config = experiments['default'].copy()
    config.update(experiments[experiment])
    experiments[experiment] = config
Export('experiments')

srtm_resolution = 3

SConscript('geography/SConscript')
SConscript('rgis/SConscript')
SConscript('upstream/SConscript')
SConscript('population/SConscript',
        exports=['srtm_resolution'])
SConscript('srtm/SConscript',
        exports=['srtm_resolution'])
SConscript('sediment/SConscript')

