# vim: set fileencoding=UTF-8 :
# vim:filetype=python

import os
import csv
import json

SetOption('max_drift', 1)

env = Environment(ENV = {'PATH' : os.environ['PATH']})
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
        'config_out': '#data/experiments/{exp}/config_{exp}.json',
        'deltas_source': ('tessler2015',
            '/Users/ztessler/data/deltas/global_map_shp/global_map.shp'),
        'deltas': '#data/Global/deltas.json',
        'delta_areas': '#data/Global/delta_areas.pd',

        'pop_dens_source': ('gpwv4',
            '/Users/ztessler/data/GPWv4_beta/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-{year}/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals_{year}.tif'),
        'pop_dens_rast': '#data/experiments/{exp}/pop_dens_{year}{ver}.{ext}',
        'delta_pop_dens': '#data/experiments/{exp}/delta_pop_dens_{year}.pd',

        'delta_map': '#data/{delta}/{delta}.json',
        'delta_pop_rast':
            '#data/gpwv4/{delta}_pop_{year}.tif',
        'delta_srtm_rast':
            '#data/srtm{srtm}/{delta}_srtm.tif',
        'delta_srtm_full_rast':
            '#data/srtm{srtm}/{delta}_srtm_full.tif',
        'delta_pop_elevations':
            '#data/{delta}/experiments/{exp}/{delta}_pop_{year}_elevations.pd',
        'delta_hypso_plot':
            '#data/{delta}/experiments/{exp}/figures/{delta}_hypsometric_{year}.png',

        'basins_source': 'rgis',
        'basins_rast': '#data/rgis/basins{ver}.{ext}',
        'reservoir_source': 'rgis',
        'reservoir_rast': '#data/rgis/reservoir{ver}.{ext}',
        'discharge_source': 'rgis',
        'discharge_rast': '#data/rgis/discharge{ver}.{ext}',
        'airtemp_source': 'rgis',
        'airtemp_rast': '#data/rgis/airtemp{ver}.{ext}',
        'ice_source': 'rgis',
        'ice_rast': '#data/rgis/ice{ver}.{ext}',
        'lithology_source': 'rgis',
        'lithology_rast': '#data/rgis/lithology{ver}.{ext}',
        'relief_source': 'rgis',
        'relief_rast': '#data/rgis/relief{ver}.{ext}',
        # 'gnp_rast': '#data/rgis/gnp{ver}.{ext}',
        'groundwater_source': ('wada', '/Users/ztessler/data/Groundwater_Wada2012/gwd02000.asc'),
        'groundwater_rast': '#data/wada/groundwater{ver}.{ext}',
        'oilgas_source': ('usgs',
            '/Users/ztessler/data/WorldPetroleumAssessment/tps_sumg/tps_sumg.shp'),
        'oilgas_vect': '#data/usgs/oilgas/oilgas.shp',

        'national_borders_source': ('HIU.State.Gov',
            '/Users/ztessler/data/HIU.State.Gov_National_Boundaries/countries.json'),
        'per_capita_gdp_source': ('worldbank',
            '/Users/ztessler/data/GDP_per_capita_WorldBank/ca0453f8-8c4c-4825-b40b-1a1bcd139c6a_v2.csv'),
        'per_capita_gdp_rast': '#data/experiments/{exp}/per_capita_gdp{ver}.{ext}',

        'basins': '#data/experiments/{exp}/basins{ver}.{ext}',
        'reservoirs': '#data/experiments/{exp}/reservoir{ver}.pd',
        'discharge': '#data/experiments/{exp}/discharge{ver}.pd',
        'airtemp': '#data/experiments/{exp}/airtemp{ver}.pd',
        'groundwater': '#data/experiments/{exp}/groundwater{ver}.pd',
        'ice': '#data/experiments/{exp}/ice{ver}.pd',
        'lithology': '#data/experiments/{exp}/lithology{ver}.pd',
        'per_capita_gdp': '#data/experiments/{exp}/per_capita_gdp{ver}.pd',
        'relief': '#data/experiments/{exp}/relief{ver}.pd',
        'pop_dens': '#data/experiments/{exp}/pop_dens_{year}{ver}.pd',

        'I': '#data/experiments/{exp}/bqart_I.pd',
        'Te': '#data/experiments/{exp}/bqart_Te.pd',
        'Eh': '#data/experiments/{exp}/bqart_Eh.pd',
        'B': '#data/experiments/{exp}/bqart_B.pd',
        'Qs': '#data/experiments/{exp}/bqart_Qs.pd',

        'groundwater_drawdown': '#data/experiments/{exp}/drawdown.pd',
        'groundwater_subsidence': '#data/experiments/{exp}/groundwater_subsidence.pd',

        'oilgas': '#data/experiments/{exp}/oilgas.pd',
        'oilgas_subsidence': '#data/experiments/{exp}/oilgas_subsidence.pd',
        'basin_ids': '#data/experiments/{exp}/basin_ids.pd',
        'basin_mouths': '#data/experiments/{exp}/basin_mouths.pd',
        'basin_areas': '#data/experiments/{exp}/basin_areas.pd',

        'natural_subsidence': '#data/experiments/pristine/natural_subsidence.pd',
        'sed_aggradation': '#data/experiments/{exp}/sed_aggradation.pd',
        'rslr': '#data/experiments/{exp}/rslr.pd',

        'basin_pixel_areas': '#data/experiments/{exp}/basin_pixel_areas.tif',
        'zeros': '#data/experiments/{exp}/zeros.pd',
        'ones': '#data/experiments/{exp}/ones.pd',

        'eustatic_slr': 3.0,
        'srtm': 3,
        }
experiments = {
        'contemp': {
            'eustatic_slr': 3.0,
            },
        'pristine': {
            'Te': defaults['zeros'],
            'Eh': defaults['ones'],
            'eustatic_slr': 1.5,
            }
        }
for experiment in experiments.keys():
    config = defaults.copy()
    config.update(experiments[experiment])
    for name, path in config.items():
        if isinstance(path, str):
            config[name] = path.format(exp=experiment, year='{year}', ver='{ver}', ext='{ext}', delta='{delta}', srtm='{srtm}')
    experiments[experiment] = config
Export('experiments')


SConscript('geography/SConscript')
SConscript('population/SConscript')
SConscript('srtm/SConscript')
SConscript('rgis/SConscript')
SConscript('upstream/SConscript')
SConscript('sediment/SConscript')
SConscript('subsidence/SConscript')

def save_config(env, target, source):
    config = env['config']
    with open(str(target[0]), 'w') as f:
        json.dump(config, f)
    return 0
for experiment, config in experiments.iteritems():
    env.Command(
            target=config['config_out'],
            source=None,
            action=save_config,
            config=config)

