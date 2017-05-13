import pandas

def analysis(env, source, target):
    contemp_rslr = pandas.read_pickle(str(source[0])).drop('Congo')
    US_util_rslr = pandas.read_pickle(str(source[1])).drop('Congo')
    retentionlow_rslr = pandas.read_pickle(str(source[2])).drop('Congo')
    USutil_retentionlow_rslr = pandas.read_pickle(str(source[3])).drop('Congo')
    natsub = pandas.read_pickle(str(source[4])).drop('Congo')
    prist_qs = pandas.read_pickle(str(source[5])).drop('Congo').groupby(level='Delta').sum()
    contemp_qs = pandas.read_pickle(str(source[6])).drop('Congo').groupby(level='Delta').sum()
    zarfl_qs = pandas.read_pickle(str(source[7])).drop('Congo').groupby(level='Delta').sum()
    usutil_qs = pandas.read_pickle(str(source[8])).drop('Congo').groupby(level='Delta').sum()

    with open(str(target[0]), 'w') as fout:
        fout.write('Contemp rslr:\n\t')
        fout.write('Min: {0} ({1})\n\t'.format(contemp_rslr.min(), contemp_rslr.argmin()))
        fout.write('Max: {0} ({1})\n\t'.format(contemp_rslr.max(), contemp_rslr.argmax()))
        fout.write('Mean: {0}\n\t'.format(contemp_rslr.mean()))
        fout.write('Mean rslr with us-utilization-rate: {}\n'.format(US_util_rslr.mean()))
        fout.write('Mean rslr with low retention: {}\n'.format(retentionlow_rslr.mean()))
        fout.write('Mean rslr with US-util and low retention: {}\n'.format(USutil_retentionlow_rslr.mean()))
        fout.write('Natural subsidence:\n\t')
        fout.write('Min: {0} ({1})\n\t'.format(natsub.min(), natsub.argmin()))
        fout.write('Max: {0} ({1})\n\t'.format(natsub.max(), natsub.argmax()))
        fout.write('Mean: {0}\n\t'.format(natsub.mean()))
        fout.write('Mississippi: {0}\n\t'.format(natsub['Mississippi']))
        fout.write('Indus: {0}\n'.format(natsub['Indus']))

        percent_change = (prist_qs - contemp_qs)/prist_qs * 100
        fout.write('Percent change in sed flux, prist to contemp:\n\tMean: {}\n\tMedian: {}\n\tIndus: {}\n'.format(percent_change.mean(), percent_change.median(), percent_change['Indus']))

        table_deltas = ['Magdalena', 'Danube', 'Amazon', 'Mekong', 'Irrawaddy', 'Hong', 'Ganges', 'Indus', 'Yangtze', 'Senegal', 'Niger', 'Amur', 'Nile', 'Shatt_el_Arab']
        change_zarfl = (contemp_qs - zarfl_qs)/contemp_qs * 100
        change_usutil = (contemp_qs - usutil_qs)/contemp_qs * 100
        fout.write('Percent change in sed flux from contemp (zarfl, us-utilization):\n')
        for d in table_deltas:
            fout.write('\t{}: {}, {}\n'.format(d, change_zarfl[d], change_usutil[d]))

        percent_change = (US_util_rslr - contemp_rslr)/contemp_rslr * 100
        df = pandas.DataFrame({'contemp':contemp_rslr, 'US-util': US_util_rslr, '%change': percent_change}, columns=['contemp', 'US-util', '%change'])
        fout.write('Greatest rslr changes from contemp to US-util:\n')
        fout.write(str(df.sort_values(by='%change', ascending=False).head(10)))
        fout.write('\n\nBasins where utilization exceeds mississippi and rslr drops:\n')
        fout.write(str(percent_change[percent_change<0].sort_values())+'\n')

        retlow_diff = retentionlow_rslr - contemp_rslr
        fout.write('Difference between contemp and retentionlow:\n')
        fout.write('Min: {0} ({1})\n\t'.format(retlow_diff.min(), retlow_diff.argmin()))
        fout.write('Max: {0} ({1})\n\t'.format(retlow_diff.max(), retlow_diff.argmax()))
        fout.write('Mean: {0}\n'.format(retlow_diff.mean()))

        comb_diff = USutil_retentionlow_rslr - contemp_rslr
        fout.write('Difference between contemp and USutil+retentionlow:\n')
        fout.write('Min: {0} ({1})\n\t'.format(comb_diff.min(), comb_diff.argmin()))
        fout.write('Max: {0} ({1})\n\t'.format(comb_diff.max(), comb_diff.argmax()))
        fout.write('Mean: {0}\n'.format(comb_diff.mean()))
    return 0
