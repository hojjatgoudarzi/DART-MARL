#!/usr/bin/env python3
"""Diagnostic-only counterfactual: uses checkpoint; does not train."""
import argparse
import json
from hr_mappo.recovery_diagnostic import diagnose_checkpoint

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/pilot_cuda.yaml')
p.add_argument('--env',default='configs/rover_12.yaml')
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seeds',nargs='+',type=int,default=[810101,810202,810303,810404,810505])
a=p.parse_args()
result=diagnose_checkpoint(a.config,a.env,a.checkpoint,a.output,a.seeds)
print(json.dumps(result,indent=2,default=str))
