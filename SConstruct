# vim: set fileencoding=UTF-8 :
# vim:filetype=python

import os
import json
from config import experiments, common

SetOption('max_drift', 1)

env = Environment(ENV = {'PATH' : os.environ['PATH']})
env.Decider('MD5-timestamp')
Export('env')

Export('experiments')
Export('common')

SConscript('geography/SConscript')
SConscript('population/SConscript')
SConscript('srtm/SConscript')
SConscript('rgis/SConscript')
SConscript('upstream/SConscript')
SConscript('sediment/SConscript')
SConscript('subsidence/SConscript')
SConscript('hazards/SConscript')
SConscript('risk/SConscript')

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

