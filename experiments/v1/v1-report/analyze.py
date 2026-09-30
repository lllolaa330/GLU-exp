from pathlib import Path
import json, re, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
sources = {'V0':root.parent.parent/'v0/add32-repeat-QgXGn4',
           'V1':root.parent/'add32-repeat-A3F35u'}
data = {}
for version, path in sources.items():
    rows = []
    for i in range(6):
        p = path/f'round-{i}'
        glu = (p/'glu.txt').read_text()
        klu = (p/'klu.txt').read_text()
        err = (p/'error.txt').read_text()
        r = {'round':i, **{k:float(v) for k,v in re.findall(r'GLU (\w+): ([\d.eE+-]+) ms',glu)}}
        r['glu_total'] = float(re.search(r'Total solve wall time: (\S+)',glu)[1])
        r['klu_total'] = float(re.search(r'Total solve wall time: (\S+)',klu)[1])
        r['error'] = float(re.search(r'relative_l2_error = (\S+)',err)[1])
        assert 'correctness = PASS' in err
        rows.append(r)
    med = {k:st.median(r[k] for r in rows[1:]) for k in rows[0] if k!='round'}
    data[version] = {'rows':rows,'medians':med,'cold_klu_over_glu':med['klu_total']/med['glu_total']}
v0, v1 = data['V0']['medians'], data['V1']['medians']
summary = {'protocol':'Round 0 excluded; all rounds 1-5 retained. Historical non-interleaved comparison; no causal speedup claim.',
           'data':data,'loop_time_reduction_percent':100*(1-v1['device_levels_wall']/v0['device_levels_wall']),
           'total_time_increase_percent':100*(v1['glu_total']/v0['glu_total']-1)}
(root/'comparison.json').write_text(json.dumps(summary,indent=2))
fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,key,title in zip(axes.flat,['glu_total','device_count_entry','device_levels_wall','error'],
                        ['Total solve wall time','Device entry wall time','Level loop wall time','Relative L2 error vs KLU']):
    for x,(version,details) in enumerate(data.items()):
        vals=[r[key] for r in details['rows'][1:]]
        color=['#4477aa','#cc6677'][x]
        ax.scatter([x+d for d in [-.10,-.05,0,.05,.10]],vals,color=color,s=35,zorder=3)
        median=details['medians'][key]
        ax.hlines(median,x-.2,x+.2,color=color,lw=2)
        ax.annotate(f'{median:.3g}' if key=='error' else f'{median:.2f}',(x+.22,median),fontsize=10,va='center')
    ax.set(title=title,xticks=[0,1],xticklabels=['V0','V1'],xlim=(-.4,1.7),ylabel='Relative error' if key=='error' else 'ms')
    if key=='error':
        ax.set_ylim(2.5e-15,2.9e-15)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    else: ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('V0 vs V1 on add32: five samples and median\nHistorical runs; identical round numbers are not paired measurements',fontsize=12)
fig.savefig(root/'v0_vs_v1.png',dpi=180)
fig.savefig(root/'v0_vs_v1.svg')
print({k:v for k,v in summary.items() if k!='data'})
print('cold ratios', {v:d['cold_klu_over_glu'] for v,d in data.items()})
