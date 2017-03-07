# vim: set fileencoding=UTF-8 :
# vim:filetype=python

import os
import json
import hashlib
from config import experiments, common

SetOption('max_drift', 1)

env = Environment(ENV = {'PATH' : os.environ['PATH'],
                         'GDAL_DATA': os.environ['GDAL_DATA'],
                        })
env.Decider('MD5-timestamp')
Export('env')

Export('experiments')
Export('common')

def myCommand(target, source, action, **kwargs):
    '''
    env.Command wrapper that forces env override arguments to be sconsign
    signature database. Wraps all extra kwargs in env.Value nodes and adds
    them to the source list, after the existing sources. Changing the extra
    arguments will cause the target to be rebuilt, as long as the data's string
    representation changes.
    '''
    def hash(v):
        # if this is changed then all targets with env overrides will be rebuilt
        return hashlib.md5(repr(v)).hexdigest()
    if not isinstance(source, list):
        source = [source]
    if None in source:
        source.remove(None)
    kwargs['nsources'] = len(source)
    source.extend([env.Value('{}={}'.format(k,hash(v))) for k,v in kwargs.iteritems()])
    return env.Command(target=target, source=source, action=action, **kwargs)
Export('myCommand')

SConscript('geography/SConscript')
SConscript('population/SConscript')
SConscript('srtm/SConscript')
SConscript('rgis/SConscript')
SConscript('upstream/SConscript')
SConscript('sediment/SConscript')
SConscript('subsidence/SConscript')
SConscript('hazards/SConscript')
SConscript('vulnerability/SConscript')
SConscript('risk/SConscript')

def save_config(env, target, source):
    config = env['config']
    with open(str(target[0]), 'w') as f:
        json.dump(config, f)
    return 0
for experiment, config in experiments.iteritems():
    myCommand(
            target=config['config_out'],
            source=None,
            action=save_config,
            config=config)

