import csv
from collections import OrderedDict


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
deltas = OrderedDict(sorted(deltas.items(), key=lambda t: t[0]))

popyears = [2010] #[2000, 2005, 2010, 2015, 2020]
forecasts = [2010, 2025, 2050, 2075, 2100]

# EXPERIMENT configs
defaults = {
        'config_out': '#data/experiments/{exp}/config_{exp}.json',
        'deltas_source': ('tessler2015',
            '/Users/ztessler/data/deltas/global_map_shp/global_map.shp'),
        'deltas': '#data/Global/deltas.json',
        'delta_areas': '#data/Global/delta_areas.pd',

        'pop_dens_source': ('gpwv4',
            '/Users/ztessler/data/GPWv4_beta/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-{popyear}/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals_{popyear}.tif'),
        'pop_dens_rast': '#data/experiments/{exp}/pop_dens_{popyear}{ver}.{ext}',
        'delta_pop_dens': '#data/experiments/{exp}/delta_pop_dens_{popyear}.pd',

        'delta_map': '#data/{delta}/{delta}.json',
        'delta_pop_rast':
            '#data/gpwv4/{delta}_pop_{popyear}.tif',
        'delta_srtm_rast':
            '#data/srtm{srtm}/{delta}_srtm.tif',
        'delta_srtm_full_rast':
            '#data/srtm{srtm}/{delta}_srtm_full.tif',
        'delta_pop_elevations':
            '#data/{delta}/experiments/{exp}/{delta}_pop_{popyear}_elevations.pd',
        'deltas_pop_elevations':
            '#data/experiments/{exp}/delta_pop_{popyear}_elevations.pd',
        'deltas_pop_elevations_forecast': # populations at different elevations given rslr forecasts
            '#data/experiments/{exp}/delta_pop_{popyear}_elevations_{forecast}.pd',
        'delta_hypso_plot':
            '#data/{delta}/experiments/{exp}/figures/{delta}_hypsometric_{popyear}.png',

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

        'zeros_rast': '#data/Global/zeros.tif',
        'ones_rast': '#data/Global/ones.tif',

        'rslr_lit_source': ('higgins2014', '/Users/ztessler/data/RSLR/higgins_rslr.csv'),
        # 'rslr_lit_source': ('higgins2014_agg', '/Users/ztessler/data/RSLR/higgins_rslr_summary.csv'),
        'rslr_lit_mean_weight': 2,
        # 'rslr_lit_source': ('syvitski2009', '/Users/ztessler/data/RSLR/syvitski_2009_rslr.csv'),
        'rslr_lit': '#data/experiments/{exp}/rslr_lit.pd',

        'sed_morph_source': ('syvitski_saito_2007', '/Users/ztessler/data/RSLR/syvitski_saito_2007_morph.csv'),
        'sed_retention': '#data/experiments/{exp}/sed_retention.pd',
        'accomodation_space': '#data/experiments/{exp}/accomodation_space.pd',
        'shape_factor': .5,

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

        'pop_growth_rate_source': ('un','http://esa.un.org/unpd/wpp/DVD/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2015_POP_F02_POPULATION_GROWTH_RATE.XLS'),
        'storm_surge_source': ('unisdr', 'http://data.hdx.rwlabs.org/dataset/87ce9e07-4914-49e6-81cc-3e4913d1ea02/resource/9d30760e-292f-4e81-9f5f-8a526977aa68/download/SS-world.zip'),
        'storm_surge_zip': '#data/unisdr/storm_surge.zip',
        'storm_surge_vect': '#data/unisdr/storm_surge/storm_surge.shp',

        'basins': '#data/experiments/{exp}/basins{ver}.{ext}',
        'reservoirs': '#data/experiments/{exp}/reservoir{ver}.pd',
        'discharge': '#data/experiments/{exp}/discharge{ver}.pd',
        'airtemp': '#data/experiments/{exp}/airtemp{ver}.pd',
        'groundwater': '#data/experiments/{exp}/groundwater{ver}.pd',
        'ice': '#data/experiments/{exp}/ice{ver}.pd',
        'lithology': '#data/experiments/{exp}/lithology{ver}.pd',
        'per_capita_gdp': '#data/experiments/{exp}/per_capita_gdp{ver}.pd',
        'relief': '#data/experiments/{exp}/relief{ver}.pd',
        'pop_dens': '#data/experiments/{exp}/pop_dens_{popyear}{ver}.pd',
        'storm_surge': '#data/experiments/{exp}/storm_surge_return_levels.pd',

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
        'rslr_regress': '#data/experiments/{exp}/rslr_regress.{ext}',

        'basin_pixel_areas': '#data/experiments/{exp}/basin_pixel_areas.tif',
        'upstream_zeros': '#data/experiments/{exp}/upstream_zeros.pd',
        'upstream_ones': '#data/experiments/{exp}/upstream_ones.pd',
        'delta_zeros': '#data/experiments/{exp}/delta_zeros.pd',
        'delta_ones': '#data/experiments/{exp}/delta_ones.pd',

        'srtm': 3,
        'eustatic_slr': 3.0,
        }

experiments = {
        'contemp': defaults,
        'pristine': {
            'parent': 'contemp',
            'Te': defaults['upstream_zeros'],
            'Eh': defaults['upstream_ones'],
            'oilgas_source': ('zeros', None),
            'groundwater_source': ('zeros', None),
            'groundwater_rast': '#data/experiments/{exp}/groundwater.tif',
            'eustatic_slr': 1.5,
            },
        }

# first fill in configs with parent values
done = False
while not done: # iterate until all parents and grandparents and great... have been filled in
    done = True
    updated_experiments = {}
    for experiment in experiments.keys():
        overrides = experiments[experiment]
        if 'parent' in overrides:
            done = False
            parent = overrides['parent']
            try:
                grandparent = experiments[parent]['parent']
            except KeyError:
                grandparent = None
            expanded = experiments[parent].copy()
            expanded.update(overrides)
            if grandparent:
                expanded['parent'] = grandparent
            else:
                del expanded['parent']
            updated_experiments[experiment] = expanded
        else:
            updated_experiments[experiment] = overrides
    experiments = updated_experiments
# then set experiment directories for output files
for experiment in experiments.keys():
    config = experiments[experiment]
    for name, path in config.items():
        try:
            config[name] = path.format(exp=experiment, popyear='{popyear}', ver='{ver}', ext='{ext}', delta='{delta}', srtm='{srtm}', forecast='{forecast}')
        except AttributeError:
            pass
    experiments[experiment] = config


