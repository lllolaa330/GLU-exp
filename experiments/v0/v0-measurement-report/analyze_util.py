from pathlib import Path
from datetime import datetime, timedelta
import csv, json, re, statistics as st
from collections import defaultdict

root = Path(__file__).resolve().parent
raw = root.parent / 'add32-util-load-SXnqQA'
start = datetime.fromisoformat((raw/'load-start.txt').read_text().strip())
end = datetime.fromisoformat((raw/'load-end.txt').read_text().strip())
devices = defaultdict(list)
for row in csv.DictReader((raw/'util.csv').open()):
    whole, micros = row['timestamp'].split('.')
    # Inferred for ht-smi 2.2.12: suffix is integer microseconds, not a
    # right-padded decimal fraction. This restores chronology on all 4 GPUs.
    time = datetime.strptime(whole, '%Y/%m/%d %H:%M:%S') + timedelta(microseconds=int(micros))
    value = float(row['utilization.GPU [%]'])
    assert 0 <= value <= 100
    devices[row['deviceId']].append((time, value))
summary = {'load_start':start.isoformat(), 'load_end':end.isoformat(),
           'load_seconds':(end-start).total_seconds(),
           'timestamp_rule':'Inferred: integer microseconds after decimal point; raw CSV unchanged.',
           'mean_definition':'Arithmetic mean of samples timestamped within inclusive load window; zeros retained.',
           'devices':{}}
for name, rows in devices.items():
    gaps = [(y[0]-x[0]).total_seconds()*1000 for x,y in zip(rows,rows[1:])]
    assert min(gaps)>0
    window = [v for t,v in rows if start<=t<=end]
    summary['devices'][name] = {'samples':len(window),'sample_mean_percent':st.mean(window),
        'peak_percent':max(window),'nonzero_samples':sum(v>0 for v in window),
        'nonzero_sample_percent':100*sum(v>0 for v in window)/len(window),
        'all_sample_gap_median_ms':st.median(gaps),'all_sample_gap_min_ms':min(gaps),'all_sample_gap_max_ms':max(gaps)}
errors = []
for i in range(1,101):
    text = (raw/f'round-{i}'/'error.txt').read_text()
    assert 'correctness = PASS' in text
    errors.append(float(re.search(r'relative_l2_error = (\S+)',text)[1]))
summary['correctness_pass_count'] = len(errors)
summary['max_relative_l2_error'] = max(errors)
(root/'v0_util_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    import sys
    if '--plot' in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = devices['GPU#0']
        x = [(t-start).total_seconds() for t,v in rows]
        y = [v for t,v in rows]
        fig, ax = plt.subplots(figsize=(10,4.5),layout='constrained')
        ax.axvspan(0,(end-start).total_seconds(),color='#eef3f8',label='100-process workload window')
        ax.plot(x,y,'o-',markersize=3,linewidth=1,color='#2864a3',label='GPU#0 samples')
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.axvline((end-start).total_seconds(),color='#777777',linestyle='--',linewidth=1)
        ax.set(xlabel='Seconds relative to workload start',ylabel='Reported GPU utilization (%)',ylim=(-.05,1.35),
               title='V0 add32: 100 separate GLU processes | ht-smi sampling')
        ax.text(.02,.91,'Within window: 100 samples | mean 0.05% | peak 1% | 5 nonzero samples',transform=ax.transAxes,fontsize=10)
        ax.legend(loc='upper left',bbox_to_anchor=(0,.83),frameon=False)
        ax.spines[['top','right']].set_visible(False)
        fig.savefig(root/'gpu_util.png',dpi=180)
        fig.savefig(root/'gpu_util.svg')
