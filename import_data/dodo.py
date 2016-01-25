import os

from import_deltas import import_deltas

build = 'build'
if not os.path.exists(build):
    os.makedirs(build)

base_delta_shp = os.path.join(os.environ['HOME'], 'data', 'deltas', 'global_map_shp/global_map.shp') 
delta_shp = os.path.join(build, 'deltas_shp', 'deltas.shp')





def task_import_deltas():
    return {
            'actions': [import_deltas],
            'file_dep': [base_delta_shp],
            'targets': [delta_shp],
            'clean': ['rm -r {}'.format(os.path.dirname(delta_shp))],
            }
