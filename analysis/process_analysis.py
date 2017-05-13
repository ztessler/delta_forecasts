import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import pandas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import defaultdict

def analysis(env, source, target):
    contemp_rslr = pandas.read_pickle(str(source[0])).drop(['Congo', 'Tone'])
    zarfl_rslr = pandas.read_pickle(str(source[1])).drop(['Congo', 'Tone'])
    US_util_rslr = pandas.read_pickle(str(source[2])).drop(['Congo', 'Tone'])
    retentionlow_rslr = pandas.read_pickle(str(source[3])).drop(['Congo', 'Tone'])
    USutil_retentionlow_rslr = pandas.read_pickle(str(source[4])).drop(['Congo', 'Tone'])
    natsub = pandas.read_pickle(str(source[5])).drop(['Congo', 'Tone'])
    prist_qs = pandas.read_pickle(str(source[6])).drop(['Congo', 'Tone']).groupby(level='Delta').sum()
    contemp_qs = pandas.read_pickle(str(source[7])).drop(['Congo', 'Tone']).groupby(level='Delta').sum()
    zarfl_qs = pandas.read_pickle(str(source[8])).drop(['Congo', 'Tone']).groupby(level='Delta').sum()
    usutil_qs = pandas.read_pickle(str(source[9])).drop(['Congo', 'Tone']).groupby(level='Delta').sum()
    area = pandas.read_pickle(str(source[10])).drop(['Congo', 'Tone'])

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
        fout.write('Sebou: {0}\n\t'.format(natsub['Sebou']))
        fout.write('Moulouya: {0}\n\t'.format(natsub['Moulouya']))
        fout.write('Indus: {0}\n\n'.format(natsub['Indus']))

        fout.write('Moulouya sed / area: {}\n'.format(prist_qs['Moulouya'] / area['Moulouya']))
        fout.write('Sebou sed / area: {}\n'.format(prist_qs['Sebou'] / area['Sebou']))
        fout.write('Han sed / area: {}\n\n'.format(prist_qs['Han'] / area['Han']))

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

        percent_change = (zarfl_rslr - contemp_rslr)/contemp_rslr * 100
        df = pandas.DataFrame({'contemp':contemp_rslr, 'zarfl': zarfl_rslr, '%change': percent_change}, columns=['contemp', 'zarfl', '%change'])
        fout.write('\n\nGreatest rslr changes from contemp to zarfl:\n')
        fout.write(str(df.sort_values(by='%change', ascending=False).head(10)))

        fout.write('\n\nBasins where utilization exceeds mississippi and rslr drops:\n')
        percent_change = (US_util_rslr - contemp_rslr)/contemp_rslr * 100
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


def plot_global_map(env, source, target):
    mouths = pandas.read_pickle(str(source[0])).groupby(level='Delta').first()
    drop = env['drop']

    mouths = mouths.drop(drop)

    dodots = True
    color = '.1'
    textcolor='.1'
    fontsize = 8
    circpts = 150
    diam = np.sqrt(circpts)
    radius = 0 #.5 * diam
    defoff = 1.25 * diam

    off = defaultdict(lambda:np.array([.25*diam, defoff]))
    off['Po'] = (0, .75*defoff)
    off['Lena'] = (0, -defoff)
    off['Nile'] = (0, -defoff)
    off['Danube'] = (-diam, -defoff)
    off['Dnieper'] = (2*diam, 2*defoff)
    off['Sebou'] = (-2.5*diam, -.5*defoff)
    off['Moulouya'] = (2*diam, -1*defoff)
    off['Mississippi'] = (2*diam, defoff)
    off['Rio_Grande'] = (-3*diam, -defoff)
    off['Grijalva'] = (-2*diam, -1.5*diam)
    off['Orinoco'] = (2*diam, defoff)
    off['Amazon'] = (-2*diam, -defoff)
    off['Sao_Francisco'] = (0, -defoff)
    off['Volta'] = (-1*diam, defoff)
    off['Niger'] = (1*diam, defoff)
    off['Congo'] = (0, -defoff)
    off['Ebro'] = (-3*diam, diam)
    off['Rhone'] = (-4*diam, 2*diam)
    off['Godavari'] = (-2.0*diam, 2.5*diam)
    off['Krishna'] = (-3*diam, -1*diam)
    off['Indus'] = (-3*diam, -1*diam)
    off['Yellow'] = (-2*diam, 2*diam)
    off['Mahakam'] = (5*diam, defoff)
    off['Mekong'] = (1*diam, -defoff)
    off['Yangtze'] = (3*diam, -defoff)
    off['Hong'] = (3*diam, -2*diam)
    off['Pearl'] = (3*diam, -1*diam)
    off['Irrawaddy'] = (1.5*diam, 2.5*diam)
    off['Chao_Phraya'] = (1.5*diam, -7*diam)
    off['Ganges'] = (-.75*diam, -4.5*diam)
    off['Mahanadi'] = (-2*diam, 5*diam)
    off['Brahmani'] = (1*diam, 3*diam)
    off['Mekong'] = (2*diam, -3.5*diam)
    off['Volga'] = (2*diam, 3*diam)
    off['Burdekin'] = (0, -defoff)
    #if not commondeltas:
    off['Limpopo'] = (diam, -defoff)


    bboxprops = dict(boxstyle='round,pad=0.15', facecolor=mpl.rcParams['axes.facecolor'], edgecolor='.5')

    mpl.style.use('ggplot')
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs}, figsize=(12, 5))
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN, facecolor='0.8', edgecolor='0.8')

    for delta, (_, _, lon, lat) in mouths.iterrows():
        if delta == 'Ganges':
            dispname = 'Ganges-\nBrahmaputra'
        elif delta == 'Shatt_el_Arab':
            dispname = 'Shatt-el-Arab'
        else:
            dispname = delta.replace('_',' ')
        if dodots:
            ax.scatter(lon, lat, s=circpts, facecolor=color, edgecolor='.5', alpha=.5, zorder=10, transform=crs)
            ax.scatter(lon, lat, s=circpts/4., facecolor=color, edgecolor='.5', alpha=1, zorder=10, transform=crs)
        offset = off[delta]
        if np.sqrt(offset[0]**2 + offset[1]**2) > (4 * radius):
            arr = ax.annotate(dispname, xy=(lon,lat), xycoords='data',
                             xytext=offset, textcoords='offset points',
                             ha='center', va='center',
                             fontsize=fontsize,
                             color=textcolor,
                             bbox=bboxprops,
                             arrowprops=dict(arrowstyle='-', fc=color, ec=color,
                                             shrinkA=0,
                                             shrinkB=0))
                                             #shrinkB=1.1*radius))
        else:
            transoffset = offset_copy(ax.transData, fig=ax.get_figure(), x=offset[0], y=offset[1], units='points')
            ax.text(lon, lat, dispname, fontsize=fontsize, ha='center', va='center', color=textcolor,
                   transform=transoffset,
                   bbox=bboxprops)

    fig.savefig(str(target[0]))
    plt.close(fig)
    return 0

