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

srtm_resolution = 3

SConscript('geography/SConscript')
SConscript('rgis/SConscript')
SConscript('upstream/SConscript')
SConscript('population/SConscript',
        exports=['srtm_resolution'])
SConscript('srtm/SConscript',
        exports=['srtm_resolution'])
SConscript('sediment/SConscript')

