from pathlib import Path
import json, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
events = json.loads(next((root.parent/'add32-trace-SzUJoU').rglob('*.json')).read_text())['traceEvents']
xs = [e for e in events if e['ph'] == 'X']
# This tracer emits epoch timestamps and durations in ns. Inter-record span
# converts to 21.71 ms, consistent with the measured 21.77 ms body wall time.
def union_ms(es):
    spans = sorted((e['ts'],e['ts']+e['dur']) for e in es)
    total = 0; end = None
    for a,b in spans:
        total += max(0,b-max(a,end if end is not None else a))
        end = max(b,end if end is not None else b)
    return total/1e6
groups = collections.defaultdict(list)
for e in xs:
    groups[(e['pid'],e['name'].split('(')[0])].append(e)
summary = [{'pid':p,'name':n,'count':len(es),'sum_ms':sum(e['dur'] for e in es)/1e6,'union_ms':union_ms(es)} for (p,n),es in groups.items()]
kernels = [e for e in xs if e['pid']==2 and e['name'].startswith('RL')]
result = {'duration_unit':'ns, inferred from epoch magnitude and wall-clock cross-check', 'groups':summary,'kernel_union_ms':union_ms(kernels),'kernel_sum_ms':sum(e['dur'] for e in kernels)/1e6}
(root/'v0_trace_summary.json').write_text(json.dumps(result,indent=2))
fig, axes = plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
for ax, pid, names, title in [(axes[0],1,['hcLaunchKernel','hcDeviceSynchronize'],'Host API duration (traced run)'),(axes[1],2,['RL','RL_onecol_factorizeCurrentCol','RL_onecol_updateSubmat','RL_onecol_cleartmpMem'],'GPU kernel duration (traced run)')]:
    labels = ['Launch (225)','Synchronize (54)'] if pid==1 else ['RL (13)','Factor (71)','Update (70)','Clear (71)']
    vals = [sum(e['dur'] for e in groups[(pid,n)])/1e6 for n in names]
    ax.barh(labels,vals,color='#4477aa' if pid==1 else '#228833')
    for i,v in enumerate(vals): ax.text(v,i,f' {v:.3f}',va='center')
    ax.set(title=title,xlabel='Sum of event durations (ms)',xlim=(0,max(vals)*1.24))
    ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Separate measurements: host and GPU overlap; do not add the panels',fontsize=12)
fig.savefig(root/'trace_breakdown.png',dpi=180)
fig.savefig(root/'trace_breakdown.svg')
print('kernel sum / union ms:',result['kernel_sum_ms'],result['kernel_union_ms'])
