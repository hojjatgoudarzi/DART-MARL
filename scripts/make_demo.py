#!/usr/bin/env python3
"""Render *actual deterministic trained-policy trajectories*, never scripted motion.

Each step uses the same update ordering as ``evaluate_allocators``. In-between
frames are visual interpolation only, not additional environment steps/actions.
This is a presentation demo, not a statistical evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from dart_marl.envs.config import load_config
from dart_marl.envs.rover import RoverEnv
from dart_marl.roles.allocator import AllocationConfig
from dart_marl.roles.coordinator import RoleCoordinator
from dart_marl.roles.evaluation import _load_actor


def rollout(env_path: Path, train_env: Path, checkpoint: Path, method: str,
            seed: int, horizon: int = 24) -> dict:
    """True learned actor + algorithmic role allocator, using paired fixed seeds."""
    import torch
    cfg = load_config(env_path)
    actor, device, provenance = _load_actor(checkpoint, train_env, cfg, "cpu")
    env = RoverEnv(cfg)
    env.reset(seed=seed)
    coord = RoleCoordinator(AllocationConfig(method=method, forecast_horizon=horizon),
                            controller_sha=provenance['checkpoint_sha256'])
    coord.reset(env)
    states = []
    def capture(reward: float, served: list[bool], events: list[dict], replan: bool):
        states.append({'step':int(env.step_index),
                       'positions':env.positions.tolist(),
                       'active':env.active.astype(int).tolist(),
                       'roles':env.roles.tolist(),
                       'reported_active':env.reported_active().astype(int).tolist(),
                       'reward':float(reward),
                       'return':float(env.cumulative_reward),
                       'served':list(map(bool,served)),
                       'completed':env.poi_completed.astype(int).tolist(),
                       'events':events,
                       'replanned':bool(replan)})
    capture(0.0,[False]*len(cfg.pois),[],False)
    while not env.is_done:
        with torch.inference_mode():
            obs=torch.as_tensor(env.observe()[None],dtype=torch.float32,device=device)
            actions,_,_=actor.act(obs,deterministic=True)
        values=actions.squeeze(0).cpu().numpy()
        values[~env.active]=0
        _,reward,terminated,truncated,info=env.step(values)
        decision=None if (terminated or truncated) else coord.maybe_reassign(env)
        capture(reward,info['poi_served'][:len(cfg.pois)],info['events_next_boundary'],
                decision is not None)
    return {'method':method,'seed':seed,'config_sha256':hashlib.sha256(env_path.read_bytes()).hexdigest(),
            'controller_sha256':provenance['checkpoint_sha256'],
            'controller_seed':provenance['training_seed'],
            'role_guidance_gain':provenance['role_guidance_gain'],
            'final_return':float(env.cumulative_reward),
            'role_replans':coord.summary()['decision_count'],
            'role_switches':coord.summary()['role_switches'],
            'states':states}


def _font(size: int, bold: bool=False):
    from PIL import ImageFont
    paths=(["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"] if bold else
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"])
    for p in paths:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def render_frame(cfg, left: dict, right: dict, k:int, alpha:float,
                 event_hold:bool=False, left_label:str='SWITCH-AWARE', right_label:str='R-CART'):
    from PIL import Image, ImageDraw
    W,H=1200,710
    im=Image.new('RGB',(W,H),'#081325')
    d=ImageDraw.Draw(im)
    white='#EBF3FF'; muted='#94AAC2'; pale='#263C55'; green='#73E6BA'; cyan='#50D5F2'
    colors=['#34D1F0','#FF8198','#FFD166','#9B8DFF','#87E7A6','#FFAB6B']
    f11=_font(12);f13=_font(14);f16=_font(16,True);f18=_font(18,True);f22=_font(22,True);f30=_font(30,True)
    d.rounded_rectangle((18,14,W-18,80),radius=15,fill='#102238',outline='#2B475E',width=2)
    d.text((40,25),'DART-MARL',font=f30,fill=white)
    d.text((252,37),'REAL TRAINED-POLICY ROVER TRAJECTORIES',font=f13,fill=cyan)
    d.text((965,35),f'STEP  {k:03d} / {cfg.horizon}',font=f18,fill=white)
    event_desc=[]
    for state in (left,right):
        for e in state.get('_recent_events', state['events']):
            label=e.get('kind','fault').replace('_',' ').upper()
            short=f'{label}  ·  R{e.get("rover",0)+1}'
            if short not in event_desc: event_desc.append(short)
    if event_desc:
        event_label=' / '.join(event_desc[:2])
    else:
        event_label='FAULT-AWARE ROLE REALLOCATION  •  SAME SEED & CONTROLLER'
    d.text((42,641),event_label,font=f16,fill='#FFB56E' if event_desc else muted)
    d.rounded_rectangle((42,671,1158,681),radius=5,fill='#263C55')
    d.rounded_rectangle((42,671,42+1116*min(1,k/cfg.horizon),681),radius=5,fill=cyan)
    d.text((43,691),'Positions shown between steps are visual interpolation only; policy actions were sampled once per simulation step.',font=f11,fill=muted)
    # Map cards: exactly identical scale on both sides.
    for index,(st,method) in enumerate([(left,left_label),(right,right_label)]):
        x0=27 if index==0 else 610; y0=100; x1=x0+563; y1=624
        d.rounded_rectangle((x0,y0,x1,y1),radius=18,fill='#102238',outline='#294760',width=2)
        d.text((x0+22,y0+15),method,font=f22,fill=white)
        d.text((x0+22,y0+47),f'TEAM RETURN   {st["return"]:6.2f}',font=f16,fill=green)
        d.text((x0+365,y0+47),f'ACTIVE   {sum(st["active"])}',font=f13,fill=muted)
        # map (sub-area 470x430 with matching aspect ratio)
        mx=x0+72; my=y0+84; size=408
        d.rounded_rectangle((mx,my,mx+size,my+size),radius=11,fill='#0B1A2D',outline='#34526C',width=2)
        for z in np.linspace(0, 1, 6):
            xx=mx+z*size; yy=my+(1-z)*size
            d.line((xx,my,xx,my+size),fill='#183047',width=1)
            d.line((mx,yy,mx+size,yy),fill='#183047',width=1)
        def coord(p): return mx+p[0]/cfg.dimensions[0]*size, my+(1-p[1]/cfg.dimensions[1])*size
        for p,poi in enumerate(cfg.pois):
            cx,cy=coord(poi.position)
            radius=poi.radius/cfg.dimensions[0]*size
            col=colors[p%len(colors)]
            d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=col,width=3)
            if st['served'][p]:
                d.ellipse((cx-radius-4,cy-radius-4,cx+radius+4,cy+radius+4),outline=green,width=3)
            d.ellipse((cx-4,cy-4,cx+4,cy+4),fill=col)
            d.text((cx-radius,cy-radius-20),f'POI {p+1}   {poi.coupling}x',font=f13,fill=col)
        positions=np.asarray(st['positions'])
        active=np.asarray(st['active'],dtype=bool)
        roles=np.asarray(st['roles'],dtype=int)
        prev=st.get('_previous_positions')
        if prev is not None:
            prior=np.asarray(prev)
            positions=prior*(1-alpha)+positions*alpha
        for i in range(len(positions)):
            if not active[i] and i>=len(cfg.rovers): continue
            px,py=coord(positions[i]); role=roles[i]
            if active[i] and 0<=role<len(cfg.pois):
                tx,ty=coord(cfg.pois[role].position)
                d.line((px,py,tx,ty),fill='#2B526B',width=2)
            fill=colors[role%len(colors)] if active[i] and role>=0 else ('#E8ECEF' if active[i] else '#52667C')
            r=10
            d.ellipse((px-r-2,py-r-2,px+r+2,py+r+2),outline='#071325',width=4)
            d.ellipse((px-r,py-r,px+r,py+r),fill=fill)
            if not active[i]:
                d.line((px-6,py-6,px+6,py+6),fill='#17273B',width=2)
                d.line((px-6,py+6,px+6,py-6),fill='#17273B',width=2)
            dx,dy=((12,-10),(12,5),(-21,-17),(-21,6))[i%4]
            d.text((px+dx,py+dy),str(i+1),font=f13,fill=white if active[i] else muted)
        d.text((x0+40,y0+508),'Rover color = assigned POI  ·  X = inactive  ·  Bright POI = serviced',font=f11,fill=muted)
    return im


def build(args):
    from PIL import Image
    env_config=Path(args.env_config)
    cfg=load_config(env_config)
    checkpoint=Path(args.checkpoint)
    train_env=Path(args.training_env_config)
    manifest=Path('examples/checkpoints/manifest.json')
    if manifest.is_file() and checkpoint.parent.resolve()==manifest.parent.resolve():
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        allowed={entry['exported_actor_sha256'] for entry in json.loads(manifest.read_text())['models']}
        if sha not in allowed: raise ValueError('example actor file checksum differs from manifest')
    method_a='switch_aware'; method_b=args.compare
    left=rollout(env_config,train_env,checkpoint,method_a,args.seed,args.forecast_horizon)
    right=rollout(env_config,train_env,checkpoint,method_b,args.seed,args.forecast_horizon)
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.trace:
        trace=Path(args.trace); trace.parent.mkdir(parents=True,exist_ok=True)
        trace.write_text(json.dumps({'format':'dart-marl-real-trajectory-v1','scenario':str(env_config),
                 'controller':str(checkpoint),'interpolation_visual_only':True,'frame_ms':args.frame_ms,
                 'interpolation':args.interpolation,'event_hold':args.event_hold,
                 'left':left,'right':right},indent=2)+'\n')
    all_frames=[]
    # Duplicate boundary frames deliberately to keep failures and role changes readable.
    for step in range(cfg.horizon+1):
        a=left['states'][step].copy(); b=right['states'][step].copy()
        for st,full in ((a,left['states']),(b,right['states'])):
            st['_recent_events']=[e for prev in full[max(0,step-3):step+1] for e in prev['events']]
        if step>0:
            a['_previous_positions']=left['states'][step-1]['positions']
            b['_previous_positions']=right['states'][step-1]['positions']
        alphas=[1.] if step==0 else np.linspace(1/args.interpolation,1,args.interpolation)
        for alpha in alphas:
            all_frames.append(render_frame(cfg,a,b,step,float(alpha),
                left_label='SWITCH-AWARE',right_label=args.compare.replace('_','-').upper()))
        if a['events'] or b['events']:
            all_frames.extend([all_frames[-1]]*args.event_hold)
    # Stable palette reduces blinking from independently quantized frames.
    palette=all_frames[0].quantize(colors=128,method=Image.Quantize.FASTOCTREE)
    pal_frames=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in all_frames]
    pal_frames[0].save(out,save_all=True,append_images=pal_frames[1:],duration=args.frame_ms,
                       loop=0,disposal=2,optimize=False)
    if args.mp4:
        if not shutil.which('ffmpeg'):
            print('ffmpeg unavailable; MP4 not generated',file=sys.stderr)
        else:
            mp4=Path(args.mp4);mp4.parent.mkdir(parents=True,exist_ok=True)
            # Pipe frames at the same visual speed; explicit yuv420p gives broad compatibility.
            p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
                '-pix_fmt','rgb24','-s','1200x710','-r',str(1000/args.frame_ms),
                '-i','pipe:0','-vcodec','libx264','-pix_fmt','yuv420p',
                '-crf','23','-movflags','+faststart',str(mp4)],stdin=subprocess.PIPE)
            try:
                for frame in all_frames: p.stdin.write(frame.tobytes())
            finally:
                p.stdin.close()
            if p.wait(): raise RuntimeError('ffmpeg failed')
    print(json.dumps({'status':'rendered_from_trained_actor','steps':cfg.horizon,
        'frames':len(all_frames),'duration_seconds':len(all_frames)*args.frame_ms/1000,
        'gif':str(out),'gif_bytes':out.stat().st_size,
        'left_return':left['final_return'],'right_return':right['final_return'],
        'left_role_replans':left['role_replans'],'right_role_replans':right['role_replans'],
        'source_checkpoint_sha256':left['controller_sha256'],
        'display_interpolation_only':True},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',default='examples/checkpoints/team12_seed_314159.pt')
    ap.add_argument('--env-config',default='configs/env/rover_team_12_showcase.yaml')
    ap.add_argument('--training-env-config',default='configs/env/rover_team_12.yaml')
    ap.add_argument('--compare',choices=['fixed','greedy','r_cart','r_cart_service','r_cart_bounded_sync','r_cart_no_sync'],default='fixed')
    ap.add_argument('--seed',type=int,default=901110)
    ap.add_argument('--forecast-horizon',type=int,default=24)
    ap.add_argument('--out',default='outputs/demo_12_vs_fixed.gif')
    ap.add_argument('--mp4',default='outputs/demo_12_vs_fixed.mp4')
    ap.add_argument('--trace',default='outputs/demo_12_vs_fixed_trace.json')
    ap.add_argument('--interpolation',type=int,default=4)
    ap.add_argument('--event-hold',type=int,default=4)
    ap.add_argument('--frame-ms',type=int,default=175)
    a=ap.parse_args()
    if a.interpolation<1 or a.interpolation>8 or a.frame_ms<40 or a.event_hold<0 or a.forecast_horizon<1: ap.error('invalid demo controls')
    build(a)
