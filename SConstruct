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
defaults = {
        'reservoir': '#data/experiments/{exp}/reservoir{ver}.{ext}',
        'discharge': '#data/experiments/{exp}/discharge{ver}.{ext}',
        'airtemp': '#data/experiments/{exp}/airtemp{ver}.{ext}',
        # 'gnp': '#data/experiments/{exp}/gnp{ver}.{ext}',
        'groundwater': '#data/experiments/{exp}/groundwater{ver}.{ext}',
        'ice': '#data/experiments/{exp}/ice{ver}.{ext}',
        'lithology': '#data/experiments/{exp}/lithology{ver}.{ext}',
        'per_capita_gdp': '#data/experiments/{exp}/per_capita_gdp{ver}.{ext}',
        'relief': '#data/experiments/{exp}/relief{ver}.{ext}',
        'pop_dens': '#data/experiments/{exp}/pop_dens_{year}{ver}.{ext}',
        'I': '#data/experiments/{exp}/bqart_I.pd',
        'Te': '#data/experiments/{exp}/bqart_Te.pd',
        'Eh': '#data/experiments/{exp}/bqart_Eh.pd',
        'B': '#data/experiments/{exp}/bqart_B.pd',
        'Qs': '#data/experiments/{exp}/bqart_Qs.pd',
        'drawdown': '#data/experiments/{exp}/drawdown.pd',
        'groundwater_subsidence': '#data/experiments/{exp}/groundwater_subsidence.pd',
        'oilgas': '#data/experiments/{exp}/oilgas.pd',
        'oilgas_subsidence': '#data/experiments/{exp}/oilgas_subsidence.pd',
        'basins': '#data/Global/basins{ver}.{ext}',
        'basin_ids': '#data/Global/basin_ids.pd',
        'basin_mouths': '#data/Global/basin_mouths.pd',
        'basin_areas': '#data/Global/basin_areas.pd',
        'natural_subsidence': '#data/Global/natural_subsidence.pd',
        'deltas': '#data/Global/deltas.json',
        'delta_areas': '#data/Global/delta_areas.pd',
        'pixel_areas_06min': '#data/Global/pixel_areas_06min.tif',
        'zeros': '#data/Global/zeros.pd',
        'ones': '#data/Global/ones.pd',
        'eustatic_slr': 3.0,
        }
experiments = {
        'shared': {},
        'contemp': {
            # 'Qs': '#data/experiments/{}/bqart_Qs.pd',
            'eustatic_slr': 3.0,
            },
        'pristine': {
            'Te': defaults['zeros'],
            'Eh': defaults['ones'],
            # 'B': '#data/experiments/{}/bqart_B.pd',
            # 'Qs': '#data/experiments/{}/bqart_Qs.pd',
            'eustatic_slr': 1.5,
            }
        }
for experiment in experiments.keys():
    config = defaults.copy()
    config.update(experiments[experiment])
    for name, path in config.items():
        if isinstance(path, str):
            config[name] = path.format(exp=experiment, year='{year}', ver='{ver}', ext='{ext}')
    experiments[experiment] = config
shared = experiments['shared']
Export('experiments')
Export('shared')

srtm_resolution = 3

SConscript('geography/SConscript')
SConscript('population/SConscript',
        exports=['srtm_resolution'])
SConscript('srtm/SConscript',
        exports=['srtm_resolution'])
SConscript('rgis/SConscript')
SConscript('upstream/SConscript')
SConscript('sediment/SConscript')
SConscript('subsidence/SConscript')

