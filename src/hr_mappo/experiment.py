"""Training, checkpoint resume, same-information evaluation and provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from dataclasses import replace

import numpy as np
import torch
import yaml
from hr_mappo.envs.config import load_config
from hr_mappo.envs.rover import RoverEnv
from .config import HRConfig
from .models import TaskActor,TeamCritic
from .rollout import trajectory
from .ppo import Learner
from .scenarios import create_scenario
from .baselines import choose, make_selector
from .metrics import physical_metrics
from .curriculum import STAGES, GateState, stage_scenario
from .physics import make_env


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cfg_read(path):
    from dataclasses import fields
    d=yaml.safe_load(Path(path).read_text())
    allowed={f.name for f in fields(HRConfig)}
    unknown=set(d)-allowed
    if unknown: raise ValueError(f'unknown keys: {unknown}')
    return HRConfig(**d)


def setup(cfg):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if cfg.deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    if str(cfg.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but this PyTorch installation cannot access an NVIDIA GPU')
    if str(cfg.device).startswith('cuda'):
        torch.set_float32_matmul_precision('high' if cfg.cuda_tf32 else 'highest')
        torch.backends.cuda.matmul.allow_tf32=cfg.cuda_tf32
        torch.backends.cudnn.allow_tf32=cfg.cuda_tf32
    device=torch.device(cfg.device)
    actor=TaskActor(cfg.hidden,cfg.actor_recurrent).to(device)
    critic=TeamCritic(cfg.hidden,cfg.critic_recurrent).to(device)
    return actor,critic,Learner(actor,critic,cfg)


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')


def dev_selection_score(mean_raw_return, mean_surplus_steps, cfg):
    """Precommitted development score; explicitly NOT heldout performance.

    Surplus is measured as actual served physical staffing above coupling.
    Distinct curriculum stages are NOT comparable: selection only occurs
    in the final robust-fault stage, evaluated on the SAME dev seed list.
    """
    return float(mean_raw_return)-cfg.best_dev_overstaff_weight*float(mean_surplus_steps)


def train(cfg, envpath, out, resume=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # A new training run must not append to an old history or overwrite a
    # checkpoint by accident. Resume is the only supported continuation path.
    if resume is None and any((out / name).exists() for name in
                              ('checkpoint.pt', 'train_history.jsonl')):
        raise FileExistsError(
            f'Output folder already contains a training run: {out}. '
            'Choose a new output folder, or explicitly pass --checkpoint to resume.'
        )
    base=load_config(envpath)
    if len(base.rovers)!=12 or base.max_agents!=14: raise ValueError('12+2 required')
    if base.max_pois!=cfg.n_pois: raise ValueError('POI capacity mismatch')
    actor,critic,learner=setup(cfg)
    print(f'[setup] device={cfg.device} seed={cfg.seed} team=12+2 scenario={cfg.scenario} '
          f'objective=team_reward + {cfg.marginal_credit_weight}*{cfg.credit_mode} '
          f'overstaff_weight={cfg.overstaff_penalty_weight} curriculum={cfg.curriculum}',flush=True)
    start=0
    gate=GateState()
    best_score=float('-inf');best_update=0
    if resume:
        data=torch.load(resume,map_location=cfg.device,weights_only=False)
        if data['config']!=cfg.dict(): raise ValueError('resume config differs')
        if data['env_sha256']!=sha(envpath): raise ValueError('resume environment differs')
        actor.load_state_dict(data['actor']);critic.load_state_dict(data['critic'])
        learner.load_state_dict(data['optimizer']);start=data['update']
        best_score=float(data.get('best_dev_score',float('-inf')))
        best_update=int(data.get('best_dev_update',0))
        if best_update and not (out/'best_checkpoint.pt').exists():
            raise ValueError('resume target lacks selected best_checkpoint.pt')
        if cfg.curriculum:
            gate=GateState(**data['curriculum_gate'])
        random.setstate(data['python_rng']);np.random.set_state(data['numpy_rng']);torch.set_rng_state(data['torch_rng'])
        if torch.cuda.is_available() and data.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(data['cuda_rng'])
    write_json(out/'manifest.json',dict(config=cfg.dict(),env_file=str(envpath),env_sha256=sha(envpath),
                                         model='task-conditioned MLP-GRU/central GRU',version=cfg.version,
                                         split='development: training uses derived seeds; final held-out seeds must be fixed before tuning',
                                         curriculum_stages=STAGES if cfg.curriculum else None,
                                         dev_seeds=list(cfg.curriculum_dev_seeds),
                                         heldout_seeds=[910101,910202,910303,910404,910505],
                                         torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),
                                         gpu_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                                         physics_backend=f'{cfg.service_attribution} physical service / CPU NumPy',
                                         neural_backend=cfg.device,amp_enabled=bool(cfg.amp and cfg.device.startswith('cuda'))))
    history=out/'train_history.jsonl'
    source_digest=hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()
    if resume and data.get('hr_source_sha256') not in (None,source_digest):
        raise ValueError('resume source mismatch: code changed; use a new run, not an invalid resume')
    for update in range(start,cfg.updates):
        if str(cfg.device).startswith('cuda'):torch.cuda.synchronize()
        t0=time.perf_counter()
        if cfg.curriculum and cfg.curriculum_mode=='fixed':
            # Same physical seed/stage exposure for matched async-vs-sync
            # experiments; avoids performance-dependent curriculum confound.
            target=min(len(STAGES)-1, (update*len(STAGES))//cfg.updates)
            if target!=gate.index:
                gate.index=target;gate.updates_in_stage=0;gate.attempts=0
        actor.eval();critic.eval()
        samples=[]
        for e in range(cfg.train_episodes_per_update):
            # Independent deterministic seed based on update+environment index; no accidental repeats.
            seed=cfg.seed+1000003*(update+1)+1009*e
            scenario=stage_scenario(base,seed,gate.name,
                distribution='v24' if cfg.scenario.startswith('coordination_v24') else 'v23' if cfg.scenario.startswith('coordination_v23') else 'v22') if cfg.curriculum else create_scenario(base,seed,cfg.scenario)
            # Only after nominal/coupling curriculum stages. Policy-conditioned
            # exposure is allowed during TRAINING, but NEVER a matched-policy
            # evaluation tape (the induced victims can differ by policy).
            use_cut=(cfg.recovery_training_fraction>0 and cfg.curriculum and
                     gate.index>=3 and (e+0.5)/cfg.train_episodes_per_update <= cfg.recovery_training_fraction)
            if use_cut:
                from .interventions import ConditionalCoalitionFaultEnv
                from dataclasses import replace as dc_replace
                env=ConditionalCoalitionFaultEnv(dc_replace(scenario,events=()),
                    intervention_step=cfg.recovery_intervention_step,mode='minimal_feasible_cut')
            else:
                env=make_env(scenario,cfg)
            samples.append(trajectory(actor,critic,env,seed,cfg))
        actor.train();critic.train()
        stats=learner.update(samples)
        gate.record_update() if cfg.curriculum else None
        if str(cfg.device).startswith('cuda'):torch.cuda.synchronize()
        record=dict(update=update+1,stage=gate.name if cfg.curriculum else cfg.scenario,environment_steps=sum(len(s['physical']) for s in samples),
                    union_boundaries=sum(s['decisions'] for s in samples),
                    agent_decisions=sum(s['agent_decisions'] for s in samples),
                    event_triggered_boundaries=sum(int('observable_change' in r['decision_reasons'].values() or 'known_task_invalid' in r['decision_reasons'].values()) for ep in samples for r in ep['records']),
                    episode_returns=[s['team_return'] for s in samples],
                    train_overstaff_penalty_total=float(sum(s['overstaff_cost'] for s in samples)),
                    train_overstaff_rover_steps=int(sum(s['overstaff_rover_steps'] for s in samples)),
                    credit_mode=cfg.credit_mode,
                    mean_return=float(np.mean([s['team_return'] for s in samples])),
                    seconds=time.perf_counter()-t0,**stats)
        physical_reports=[physical_metrics(ep['physical'],
            stage_scenario(base,ep['seed'],gate.name,distribution='v24' if cfg.scenario.startswith('coordination_v24') else 'v23' if cfg.scenario.startswith('coordination_v23') else 'v22')
            if cfg.curriculum else create_scenario(base,ep['seed'],cfg.scenario)) for ep in samples]
        record['train_idle_fraction']=float(np.mean([x['idle_fraction'] for x in physical_reports]))
        record['train_surplus_per_episode']=float(np.mean([x['total_surplus_contributor_steps'] for x in physical_reports]))
        record['train_concurrent_repeatable_pois']=float(np.mean([x['mean_concurrent_repeatable_pois'] for x in physical_reports]))
        record['train_steps_two_plus_tasks']=float(np.mean([x['steps_with_2plus_repeatable_pois'] for x in physical_reports]))
        record['train_selection_counts']={str(i):sum(r['selected_task_counts'].get(str(i),0)
            for r in physical_reports) for i in range(-1,cfg.n_pois)}
        record['train_coupled_service_rate']=float(np.mean([x['coupled_service_rate'] for x in physical_reports]))
        record['train_recovery_active']=sum(r['recovery_active_reallocation'] for r in physical_reports)
        record['train_recovery_natural']=sum(r['recovery_natural_reentry'] for r in physical_reports)
        record['train_recovery_censored']=sum(r['recovery_censored'] for r in physical_reports)
        record['training_conditional_cut_episodes']=sum(s.get('intervention_status') is not None for s in samples)
        record['training_conditional_cut_opportunities']=sum(bool((s.get('intervention_status') or {}).get('opportunity')) for s in samples)
        record['training_conditional_cut_actual_drops']=sum(bool((s.get('intervention_status') or {}).get('opportunity')) and
            physical_reports[i]['recovery_opportunities']>0 for i,s in enumerate(samples))
        record['environment_steps_per_sec']=record['environment_steps']/max(record['seconds'],1e-9)
        record['agent_decisions_per_sec']=record['agent_decisions']/max(record['seconds'],1e-9)
        if cfg.curriculum and (update+1) % cfg.curriculum_eval_every == 0:
            from .baselines import choose, make_selector
            actor.eval();critic.eval()
            actor_dev=[];random_dev=[];local_dev=[];near_dev=[];coal_dev=[];idle_dev=[];fixed_dev=[];window_dev=[];static_dev=[]
            dev_coupled=[];dev_recovery=[];dev_surplus=[];dev_concurrency=[]
            for dev_seed in cfg.curriculum_dev_seeds:
                scenario=stage_scenario(base,dev_seed,gate.name,
                    distribution='v24' if cfg.scenario.startswith('coordination_v24') else 'v23' if cfg.scenario.startswith('coordination_v23') else 'v22')
                epdev=trajectory(actor,critic,make_env(scenario,cfg),dev_seed,cfg,deterministic=True)
                actor_dev.append(epdev['team_return'])
                dm=physical_metrics(epdev['physical'],scenario)
                dev_coupled.append(dm['coupled_service_rate'])
                dev_surplus.append(dm['total_surplus_contributor_steps'])
                dev_concurrency.append(dm['mean_concurrent_repeatable_pois'])
                dev_recovery.extend(dm['recovery_events'])
                rng=np.random.default_rng(dev_seed+74003)
                selection=lambda obs,env,prev:choose(obs,prev,'random',rng)
                random_dev.append(trajectory(actor,critic,make_env(scenario,cfg),dev_seed,cfg,
                                             deterministic=True,selector=selection)['team_return'])
                if cfg.curriculum_require_local:
                    idle_dev.append(trajectory(actor,critic,make_env(scenario,cfg),dev_seed,cfg,
                            deterministic=True,selector=lambda obs,env,prev:choose(obs,prev,'always_idle'))['team_return'])
                    for method,output in ([('persistent',local_dev),('nearest',near_dev),
                                          ('coalition_local',coal_dev)] +
                                         ([('fixed_initial_value',fixed_dev),('static_highest_value_repeatable',static_dev),('window_aware',window_dev)]
                                          if cfg.scenario.startswith('coordination_v24') else [])):
                        local_selector=make_selector(method)
                        output.append(trajectory(actor,critic,make_env(scenario,cfg),dev_seed,cfg,
                                            deterministic=True,selector=local_selector)['team_return'])
            record['gate_dev_actor_mean']=float(np.mean(actor_dev))
            record['gate_dev_random_mean']=float(np.mean(random_dev))
            record['gate_dev_coupled_rate']=float(np.mean(dev_coupled))
            record['gate_dev_surplus_rover_steps']=float(np.mean(dev_surplus))
            record['gate_dev_concurrent_repeatable_pois']=float(np.mean(dev_concurrency))
            record['gate_dev_selection_score']=dev_selection_score(record['gate_dev_actor_mean'],
                                              record['gate_dev_surplus_rover_steps'],cfg)
            record['selection_eligible']=bool(cfg.best_dev_checkpoint and gate.name==STAGES[-1])
            record['best_dev_selected']=False
            if record['selection_eligible'] and record['gate_dev_selection_score']>best_score:
                best_score=record['gate_dev_selection_score']
                best_update=update+1
                record['best_dev_selected']=True
            record['gate_dev_recovery_events']=len(dev_recovery)
            record['gate_dev_recovered']=sum(int(x['recovered']) for x in dev_recovery)
            record['gate_dev_natural_reentry']=sum(x.get('recovery_type')=='natural_reentry' for x in dev_recovery)
            record['gate_dev_active_reallocation']=sum(x.get('recovery_type') in
                ('active_coalition_reallocation','replacement_contribution') for x in dev_recovery)
            if cfg.scenario.startswith(('coordination_v23','coordination_v24')):
                # A rover returning after a fixed outage is not evidence of
                # learned task transfer. Gate 4/5 requires *active* repair.
                record['gate_dev_recovered']=record['gate_dev_active_reallocation']
            record['gate_dev_recovery_rate']=(record['gate_dev_recovered']/len(dev_recovery)
                    if dev_recovery else None)
            if cfg.curriculum_require_local:
                record['gate_dev_idle_mean']=float(np.mean(idle_dev))
                record['gate_dev_persistent_mean']=float(np.mean(local_dev))
                record['gate_dev_nearest_mean']=float(np.mean(near_dev))
                record['gate_dev_coalition_mean']=float(np.mean(coal_dev))
                if fixed_dev: record['gate_dev_fixed_mean']=float(np.mean(fixed_dev))
                if window_dev: record['gate_dev_window_mean']=float(np.mean(window_dev))
                if static_dev: record['gate_dev_static_highest_mean']=float(np.mean(static_dev))
            local_floor=max([float(np.mean(x)) for x in (local_dev,near_dev,coal_dev,fixed_dev,window_dev,static_dev) if x],default=0.)
            floor=max(record['gate_dev_random_mean']+cfg.curriculum_margin_vs_random,
                      float(np.mean(idle_dev))+cfg.curriculum_margin_vs_random if idle_dev else 0.,
                      cfg.curriculum_local_fraction*local_floor if cfg.curriculum_require_local else 0.)
            stage_idx=gate.index
            requires_coupling=(stage_idx >= 1)
            requires_recovery=(stage_idx >= 3)
            evidence_ok=(not requires_coupling or record['gate_dev_coupled_rate']>=cfg.curriculum_min_coupled_rate)
            if requires_recovery:
                evidence_ok=evidence_ok and (len(dev_recovery)>=cfg.curriculum_min_recovery_events and
                           (record['gate_dev_recovery_rate'] or 0.)>=cfg.curriculum_min_recovery_rate)
            record['gate_required_floor']=floor
            record['gate_evidence_ok']=bool(evidence_ok)
            if cfg.curriculum_mode=='gated':
                record['gate_passed']=gate.assess(record['gate_dev_actor_mean'] if evidence_ok else float('-inf'),
                           floor,cfg.curriculum_min_updates,0.)
            else:
                gate.attempts+=1
                record['gate_passed']=bool(evidence_ok and record['gate_dev_actor_mean']>=floor)
            record['next_stage']=gate.name
        with history.open('a') as fp: fp.write(json.dumps(record)+'\n')
        checkpoint=dict(actor=actor.state_dict(),critic=critic.state_dict(),
                        optimizer=learner.state_dict(),config=cfg.dict(),env_sha256=sha(envpath),
                        update=update+1,curriculum_gate=gate.__dict__.copy(),
                        best_dev_score=best_score,best_dev_update=best_update,
                        hr_source_sha256=source_digest,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        tmp=out/'checkpoint.tmp';torch.save(checkpoint,tmp);tmp.replace(out/'checkpoint.pt')
        if record.get('best_dev_selected',False):
            # Full resumable checkpoint; not just actor weights. The update
            # is selected only from robust-stage development seeds.
            import shutil
            shutil.copyfile(out/'checkpoint.pt',out/'best_checkpoint.pt')
            torch.save(actor.state_dict(),out/'best_actor_only.pt')
            write_json(out/'best_development.json',dict(
                update=best_update,score=best_score,
                raw_return=record['gate_dev_actor_mean'],
                surplus_rover_steps=record['gate_dev_surplus_rover_steps'],
                concurrent_repeatable_pois=record['gate_dev_concurrent_repeatable_pois'],
                selection_rule='dev mean raw return minus coefficient times mean physical surplus rover-steps',
                overstaff_coefficient=cfg.best_dev_overstaff_weight,
                scenario_stage=gate.name,seeds=list(cfg.curriculum_dev_seeds),
                certified_active_recovery=bool(record.get('gate_dev_recovery_events',0)>0
                    and record.get('gate_dev_active_reallocation',0)>0),
                warning='Development-selected; no heldout use and no recovery claim without actual outage'))
        if (cfg.curriculum and cfg.curriculum_mode=='gated' and cfg.curriculum_max_stall_updates and
            gate.updates_in_stage >= cfg.curriculum_max_stall_updates and gate.index < len(STAGES)-1):
            (out/'CURRICULUM_GATE_FAILED.txt').write_text(
                f'At update {update+1}, stage {gate.name} still failed validation. '
                'Stop large runs; inspect learning and task distribution.\n')
            raise RuntimeError('Curriculum gate failed; training stopped to avoid wasted compute')
        print(f"[train] {update+1:04d}/{cfg.updates} stage={record['stage']} "
              f"return={record['mean_return']:.2f} entropy={stats['entropy']:.3f} "
              f"surplus={record['train_surplus_per_episode']:.1f} "
              f"active_recovery={record['train_recovery_active']} "
              f"decisions={record['agent_decisions']} steps/sec={record['environment_steps_per_sec']:.1f} "
              f"seconds={record['seconds']:.1f}",flush=True)
    torch.save(actor.state_dict(),out/'actor_only.pt')
    final_stage_train_updates=sum(json.loads(line)['stage']==STAGES[-1] for line in history.read_text().splitlines()) if cfg.curriculum else None
    write_json(out/'research_readiness.json',dict(
        complete_updates=cfg.updates,final_stage=gate.name if cfg.curriculum else None,
        final_stage_training_updates=final_stage_train_updates,
        best_dev_update=best_update,best_dev_score=best_score if best_update else None,
        best_checkpoint_exists=(out/'best_checkpoint.pt').is_file(),
        recommended_for_final_claims=False,
        message='Only independent training seeds and controlled held-out evaluation can establish a coordination/recovery claim.'))


def evaluate(cfg,envpath,checkpoint,out,seeds,method='actor',trace=False):
    base=load_config(envpath)
    actor,critic,_=setup(cfg)
    if checkpoint and method!='untrained':
        data=torch.load(checkpoint,map_location=cfg.device,weights_only=False)
        if 'config' in data:
            immutable=('hidden','actor_recurrent','critic_recurrent','n_agents','n_pois',
                       'decision_interval','decision_mode','async_min_interval',
                       'communication','message_delay','message_dropout',
                       'message_ttl','message_range','reset_on_reentry','service_attribution','value_output_scale',
                       'marginal_credit_weight','idle_decision_penalty','potential_shaping_weight',
                       'credit_mode','overstaff_penalty_weight')
            mismatch=[k for k in immutable if data['config'].get(k)!=getattr(cfg,k)]
            if mismatch:
                raise ValueError(f'evaluation differs from training: {mismatch}')
        actor.load_state_dict(data['actor'] if 'actor' in data else data)
        if 'critic' in data: critic.load_state_dict(data['critic'])
    elif method=='actor':
        raise ValueError('trained actor evaluation requires checkpoint')
    actor.eval();critic.eval()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        scenario=create_scenario(base,seed,cfg.scenario)
        rng=np.random.default_rng(seed+88001)
        select=None if method in ('actor','untrained') else make_selector(method,rng)
        sample=trajectory(actor,critic,make_env(scenario,cfg),seed,cfg,True,select)
        metric=physical_metrics(sample['physical'],scenario)
        write_json(out/f'{method}_seed_{seed}_metrics.json',metric)
        row=dict(method=method,seed=seed,return_raw=sample['team_return'],
                 steps=len(sample['physical']),scenario=cfg.scenario)
        results.append(row)
        if trace: write_json(out/f'{method}_seed_{seed}_trace.json',sample['physical'])
        print(row,flush=True)
    with (out/f'{method}_episodes.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    write_json(out/f'{method}_summary.json',dict(method=method,mean_return=float(np.mean([r['return_raw'] for r in results])),
                                                   std=float(np.std([r['return_raw'] for r in results],ddof=1)) if len(results)>1 else None,
                                                   count=len(results),seeds=seeds))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['train','evaluate'])
    ap.add_argument('--config',required=True)
    ap.add_argument('--env',default='configs/rover_12.yaml')
    ap.add_argument('--output',required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--method',choices=['actor','untrained','random','nearest','value_travel','persistent','coalition_local','always_idle','static_highest_value_repeatable','fixed_initial_value','fixed_initial_nearest','window_aware'],default='actor')
    ap.add_argument('--scenario',choices=['legacy','random_nominal','random_faults','heterogeneous','stress',
                                           'recovery_challenge','recovery_heldout','recovery_nominal',
                                           'coordination_v23','coordination_v23_nominal',
                                           'coordination_v23_heldout','coordination_v23_stress',
                                           'coordination_v24','coordination_v24_nominal',
                                           'coordination_v24_heldout','coordination_v24_stress'],default=None)
    ap.add_argument('--device',choices=['cpu','cuda'],default=None)
    ap.add_argument('--seeds',type=int,nargs='+',default=[900101,900202,900303,900404,900505])
    ap.add_argument('--trace',action='store_true')
    args=ap.parse_args();cfg=cfg_read(args.config)
    if args.device is not None:cfg=replace(cfg,device=args.device)
    if args.scenario is not None:
        if args.mode=='train' and cfg.curriculum:
            raise ValueError('cannot override staged curriculum with a fixed scenario')
        cfg=replace(cfg,scenario=args.scenario)
    if args.mode=='train': train(cfg,args.env,args.output,args.checkpoint)
    else: evaluate(cfg,args.env,args.checkpoint,args.output,args.seeds,args.method,args.trace)

if __name__=='__main__': main()
