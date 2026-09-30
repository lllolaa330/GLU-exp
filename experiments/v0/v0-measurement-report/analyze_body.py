from pathlib import Path
import re, json, csv, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
raw = root.parent / 'add32-repeat-QgXGn4'
keys = ['device_events_alloc', 'device_h2d_structure', 'device_h2d_values',
        'device_workspace', 'device_host_numeric_prepare', 'device_levels_wall',
        'device_factors_d2h', 'device_finalize']
rows = []
for i in range(6):
    p = raw / f'round-{i}'
    glu = (p / 'glu.txt').read_text()
    klu = (p / 'klu.txt').read_text()
    err = (p / 'error.txt').read_text()
    r = {'round': i, **{k: float(v) for k, v in re.findall(r'GLU (\w+): ([\d.eE+-]+) ms', glu)}}
    r['glu_total'] = float(re.search(r'Total solve wall time: ([\d.eE+-]+)', glu)[1])
    r['klu_total'] = float(re.search(r'Total solve wall time: ([\d.eE+-]+)', klu)[1])
    r['relative_l2_error'] = float(re.search(r'relative_l2_error = ([\d.eE+-]+)', err)[1])
    assert 'correctness = PASS' in err
    r['body_sum_difference_ms'] = sum(r[k] for k in keys) - r['device_body_mixed']
    assert abs(r['body_sum_difference_ms']) < 1e-5
    rows.append(r)
with (root / 'samples.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
formal = rows[1:]
med = {k: st.median(r[k] for r in formal) for k in rows[0] if k != 'round'}
summary = {'formal_rounds': [1,2,3,4,5], 'median_ms_except_error': med,
           'cold_speedup_ratio_of_medians': med['klu_total']/med['glu_total'],
           'median_per_run_loop_body_fraction': st.median(r['device_levels_wall']/r['device_body_mixed'] for r in formal),
           'max_body_sum_difference_ms': max(abs(r['body_sum_difference_ms']) for r in rows)}
(root / 'v0_device_body_repeat_summary.json').write_text(json.dumps(summary, indent=2))
fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
groups = [('Events + buffers', ['device_events_alloc']), ('Structure + values H2D', ['device_h2d_structure','device_h2d_values']), ('Workspace', ['device_workspace']), ('Host numeric preparation', ['device_host_numeric_prepare']), ('Level loop (wall)', ['device_levels_wall']), ('Factors D2H', ['device_factors_d2h']), ('Finalize', ['device_finalize'])]
bottom = [0.0]*5
for label, fields in groups:
    vals = [sum(r[k] for k in fields) for r in formal]
    ax.bar(range(1,6), vals, bottom=bottom, label=label)
    bottom = [a+b for a,b in zip(bottom, vals)]
for x, y in zip(range(1,6), bottom): ax.text(x,y+.25,f'{y:.2f}',ha='center',fontsize=10)
ax.set(xlabel='Round (round 0 excluded)', ylabel='Host wall time (ms)', title='V0 add32: device body breakdown', ylim=(0,28), xticks=range(1,6))
ax.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'body_stages.png', dpi=180)
fig.savefig(root/'body_stages.svg')
print(json.dumps(summary,indent=2))
