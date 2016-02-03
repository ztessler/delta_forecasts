# vim: set fileencoding=UTF-8 :
# vim:filetype=python

import os

SetOption('max_drift', 1)

env = Environment(ENV=os.environ)
env.Decider('MD5-timestamp')
Export('env')

SConscript('load_data/SConscript')
SConscript('population/SConscript')

Clean('.', 'data')
