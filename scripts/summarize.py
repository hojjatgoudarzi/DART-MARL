#!/usr/bin/env python3
"""Make a compact, honest index of real evaluation files (no invented scores)."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
p.add_argument('--output',required=True)
a=p.parse_args(); root=Path(a.run)
out={'classification':'development, not blind held-out', 'methods':{},'stress':{}}
for file in sorted((root/'matched').glob('*_episodes.csv')):
    rows=list(csv.DictReader(file.open()))
    if not rows: continue
    vals=[float(r['return_raw']) for r in rows]
    out['methods'][rows[0]['method']]={'episodes':len(rows),'mean_return':sum(vals)/len(vals)}
file=root/'stress_cut/intervention_summary.json'
if file.exists():out['stress']=json.loads(file.read_text())
p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'method_means':out['methods'],'stress_summary':str(file)},indent=2))
