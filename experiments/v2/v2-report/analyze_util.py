from pathlib import Path
from datetime import datetime, timedelta
import csv, json, re, statistics as st
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--v0', type=Path, default=Path(__file__).resolve().parent.parent.parent/'v0/add32-util-load-SXnqQA')
parser.add_argument('--v1', type=Path, default=Path(__file__).resolve().parent.parent.parent/'v1/add32-util-load-hHXvh0')
parser.add_argument('--v2', type=Path, default=Path(__file__).resolve().parent.parent/'add32-util-load-BMyNGn')
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
results, curves = {}, {}
for version, raw in [('V0',args.v0),('V1',args.v1),('V2',args.v2)]:
    start = datetime.fromisoformat((raw/'load-start.txt').read_text().strip())
    end = datetime.fromisoformat((raw/'load-end.txt').read_text().strip())
    devices = defaultdict(list)
    for row in csv.DictReader((raw/'util.csv').open()):
        whole, micros = row['timestamp'].split('.')
        # Same inferred ht-smi format as V0: integer microseconds suffix.
        t = datetime.strptime(whole,'%Y/%m/%d %H:%M:%S') + timedelta(microseconds=int(micros))
        value = float(row['utilization.GPU [%]'])
        assert 0 <= value <= 100
        devices[row['deviceId']].append((t,value))
    summary = {'load_start':start.isoformat(),'load_end':end.isoformat(),'load_seconds':(end-start).total_seconds(),'devices':{}}
    for device, rows in devices.items():
        gaps=[(b[0]-a[0]).total_seconds()*1000 for a,b in zip(rows,rows[1:])]
        assert min(gaps)>0, (version,device,'timestamp order')
        values=[v for t,v in rows if start<=t<=end]
        assert values
        summary['devices'][device]={'samples':len(values),'sample_mean_percent':st.mean(values),
            'peak_percent':max(values),'nonzero_samples':sum(v>0 for v in values),
            'nonzero_sample_percent':100*sum(v>0 for v in values)/len(values),
            'all_sample_gap_median_ms':st.median(gaps),'all_sample_gap_min_ms':min(gaps),'all_sample_gap_max_ms':max(gaps)}
    errors=[]
    for i in range(1,101):
        text=(raw/f'round-{i}'/'error.txt').read_text()
        assert 'correctness = PASS' in text
        errors.append(float(re.search(r'relative_l2_error = (\S+)',text)[1]))
    summary['pass_count']=len(errors)
    summary['max_relative_l2_error']=max(errors)
    results[version]=summary
    curves[version]=([(t-start).total_seconds() for t,v in devices['GPU#0']], [v for t,v in devices['GPU#0']])
(root/'util_comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(3,1,figsize=(10,8),layout='constrained',sharex=True,sharey=True)
    peak=max(max(y) for x,y in curves.values())
    for ax,(version,summary),color in zip(axes,results.items(),['#4477aa','#cc6677','#228833']):
        d=summary['devices']['GPU#0']; duration=summary['load_seconds']
        ax.axvspan(0,duration,color=color,alpha=.07)
        ax.plot(*curves[version],marker='o',markersize=3,lw=1,color=color)
        ax.axvline(0,color='#666666',ls='--',lw=1)
        ax.axvline(duration,color='#666666',ls='--',lw=1)
        ax.set(title=f"{version}: {duration:.2f} s | mean {d['sample_mean_percent']:.3f}% | peak {d['peak_percent']:g}% | nonzero {d['nonzero_samples']}/{d['samples']}",
               ylabel='Reported GPU util (%)',ylim=(-.05,max(peak*1.2,1.2)))
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Seconds relative to each workload start')
    fig.suptitle('GPU#0: 100 separate GLU processes per version\nShading marks workload window; mean includes zero samples',fontsize=12)
    fig.savefig(root/'v0_v1_v2_util.png',dpi=180)
    fig.savefig(root/'v0_v1_v2_util.svg')
