import csv
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
