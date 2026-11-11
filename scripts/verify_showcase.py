#!/usr/bin/env python3
"""Verify that a 12-rover showcase is an unmodified, paired trained-policy rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dart_marl.envs.config import load_config


def verify(trace_path: Path, env_path: Path, checkpoint: Path, gif: Path | None = None,
           compare: str = "r_cart") -> dict:
    cfg=load_config(env_path)
    trace=json.loads(trace_path.read_text())
    assert trace['interpolation_visual_only'] is True
    assert len(cfg.rovers)==12 and cfg.max_agents>=14
    assert hashlib.sha256(env_path.read_bytes()).hexdigest()==trace['left']['config_sha256']
    expected=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert expected==trace['left']['controller_sha256']==trace['right']['controller_sha256']
    left,right=trace['left'],trace['right']
    assert left['seed']==right['seed']
    assert compare in ('r_cart','fixed','greedy','r_cart_service','r_cart_bounded_sync','r_cart_no_sync')
    assert (left['method'],right['method'])==('switch_aware',compare)
    assert len(left['states'])==len(right['states'])==cfg.horizon+1
    assert np.allclose(left['states'][0]['positions'],right['states'][0]['positions'])
    assert int(sum(left['states'][0]['active']))==12
    event_sets=[]
    for side in (left,right):
        assert np.isfinite(side['final_return'])
        states=side['states']
        assert all(s['step']==idx for idx,s in enumerate(states))
        assert all(np.isfinite(np.asarray(s['positions'],dtype=float)).all() for s in states)
        assert all(len(s['positions'])==cfg.max_agents for s in states)
        assert np.isclose(states[-1]['return'],side['final_return'])
        event_sets.append([(s['step'],e['kind'],e['rover']) for s in states for e in s['events']])
    assert event_sets[0]==event_sets[1], 'fault tapes must match exactly'
    replacement=[e for e in event_sets[0] if e[1]=='replacement_agent']
    reentries=[e for e in event_sets[0] if e[1]=='automatic_reentry']
    assert len(replacement)==2 and len(reentries)==3
    duration=None
    if gif is not None:
        from PIL import Image
        im=Image.open(gif)
        duration=0
        for idx in range(im.n_frames):
            im.seek(idx)
            duration+=im.info.get('duration',0)
        duration/=1000
        if not 75<=duration<=110:
            raise ValueError(f'expected a near-90-second showcase, got {duration:.1f}s')
    return {'status':'verified','initial_rovers':12,'reserved_slots':cfg.max_agents-12,
            'steps':cfg.horizon,'recorded_events':len(event_sets[0]),
            'replacements':len(replacement),'reentries':len(reentries),
            'return_switch_aware':left['final_return'],f'return_{compare}':right['final_return'],
            'checkpoint_sha256':expected,'gif_duration_seconds':duration,
            'note':'No claim that the controller was trained for 12 rovers is made by this verifier; pass a checkpoint actually trained on the 12-rover config.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--env-config',type=Path,default=Path('configs/env/rover_team_12_showcase.yaml'))
    ap.add_argument('--gif',type=Path)
    ap.add_argument('--compare',choices=['r_cart','fixed','greedy','r_cart_service','r_cart_bounded_sync','r_cart_no_sync'],default='r_cart')
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    info=verify(a.trace,a.env_config,a.checkpoint,a.gif,a.compare)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
