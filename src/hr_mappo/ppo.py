"""Masked, vectorized on-policy recurrent PPO with optional truncated BPTT.

Pads *whole independent episodes*, never random individual transitions.
A B x T x N rollout becomes B*N batched agent GRUs on CUDA. Nondecision
agents never receive an actor loss or a hidden-state update. Value remains
one team value per physical-union decision boundary, independent of N.
Chunked training reconstructs recurrent state under CURRENT network weights;
burn-in replay is gradient-free, never treated as an independent transition.
"""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from .rollout import tensorize


class Learner:
    def __init__(self,actor,critic,cfg):
        self.actor=actor;self.critic=critic;self.cfg=cfg
        self.actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg.lr,eps=1e-5)
        self.critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg.lr,eps=1e-5)
        self.amp=bool(cfg.amp and next(actor.parameters()).device.type=='cuda')
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.amp)
        self.entropy_coef=float(cfg.entropy)

    def _pack(self, episodes, device):
        b=len(episodes)
        T=max(len(ep['records']) for ep in episodes)
        template=episodes[0]['records'][0]
        op={};states={}
        for name in template['obs']:
            src=np.asarray(template['obs'][name]);data=np.zeros((b,T,*src.shape),dtype=src.dtype)
            if name=='legal':data[..., -1]=True
            for k,ep in enumerate(episodes):
                for t,r in enumerate(ep['records']):data[k,t]=r['obs'][name]
            op[name]=torch.as_tensor(data,device=device)
        for name in template['state']:
            src=np.asarray(template['state'][name]);data=np.zeros((b,T,*src.shape),dtype=src.dtype)
            for k,ep in enumerate(episodes):
                for t,r in enumerate(ep['records']):data[k,t]=r['state'][name]
            states[name]=torch.as_tensor(data,device=device)
        extras={}
        specs=dict(actions=(self.cfg.n_agents,),old_logp=(self.cfg.n_agents,),
                   decide=(self.cfg.n_agents,),memory_keep=(self.cfg.n_agents,),
                   advantage=(),marginal_return=(self.cfg.n_agents,),**{'return':()})
        for field,shape in specs.items():
            if field in ('decide','memory_keep'):dtype=bool
            elif field=='actions':dtype=np.int64
            else:dtype=np.float32
            arr=np.zeros((b,T,*shape),dtype=dtype)
            if field=='actions':arr.fill(self.cfg.n_pois)  # valid IDLE for pads
            if field=='memory_keep':arr.fill(True)
            for k,ep in enumerate(episodes):
                for t,r in enumerate(ep['records']):
                    if field=='decide' and field not in r:arr[k,t]=r['active']
                    else:arr[k,t]=r[field]
            extras[field]=torch.as_tensor(arr,device=device)
        time_valid=torch.zeros((b,T),dtype=torch.bool,device=device)
        for k,ep in enumerate(episodes):time_valid[k,:len(ep['records'])]=True
        return op,states,extras,time_valid,T

    def _step(self, packed, t, h, vh, with_grad=True):
        obs,states,ex,valid,_=packed
        B,N=obs['active'].shape[0],obs['active'].shape[2]
        sliced={k:v[:,t].reshape(B*N,*v.shape[3:]) for k,v in obs.items()}
        s={k:v[:,t] for k,v in states.items()}
        dist,proposal=self.actor.distribution(sliced,h.reshape(B*N,-1))
        # No hidden updates for a rover that did not choose at this boundary.
        due=ex['decide'][:,t] & valid[:,t,None]
        ah=torch.where(due.reshape(B*N,1),proposal,h.reshape(B*N,-1))
        vh0,crit_h=self.critic.forward_step(s,vh)
        vh=torch.where(valid[:,t,None],crit_h,vh)
        keep=ex['memory_keep'][:,t] | ~valid[:,t,None]
        next_h=ah.reshape(B,N,-1)*keep.unsqueeze(-1)
        return dist,vh0,next_h,vh

    def update(self,episodes):
        if any(ep.get('on_policy_compatible',True) is False for ep in episodes):
            raise ValueError('cannot PPO-update from classical evaluation trajectories')
        device=next(self.actor.parameters()).device
        packed=self._pack(episodes,device)
        obs,states,ex,valid,T=packed
        B,N=obs['active'].shape[0],self.cfg.n_agents
        # Shared team GAE appears exactly ONCE in critic loss, not N times.
        all_adv=ex['advantage'][valid]
        mean=all_adv.mean();std=all_adv.std(unbiased=False).clamp_min(1e-6)
        # Per-agent physical credit is an OPTIONAL PPO advantage component.
        # Pivotal (legacy) and threshold-game Shapley (v2.5) are distinct;
        # neither is future-aware D++ nor an execution-time actor input.
        raw_credit=ex['marginal_return']
        credit_mask=ex['decide'] & obs['active'] & valid[:,:,None]
        if bool(credit_mask.any()):
            samples=raw_credit[credit_mask]
            cmean=samples.mean();cstd=samples.std(unbiased=False).clamp_min(1e-6)
        else:
            cmean=raw_credit.new_zeros(());cstd=raw_credit.new_ones(())
        chunk=self.cfg.chunk_length if self.cfg.chunk_length else T
        metrics=[]
        entropy_ratios=[]
        for epoch in range(self.cfg.epochs):
            for start in range(0,T,chunk):
                end=min(start+chunk,T)
                h=torch.zeros(B,N,self.cfg.hidden,device=device)
                vh=torch.zeros(B,self.cfg.hidden,device=device)
                # Prefix is rebuilt under CURRENT weights, never reuses
                # stale acting hidden states from an older PPO iteration.
                # Burn-in [start-burn_in,start) is explicitly replayed without
                # gradients after its earlier prefix reconstruction.
                pre=max(0,start-self.cfg.burn_in)
                with torch.no_grad():
                    for t in range(pre):
                        _,_,h,vh=self._step(packed,t,h,vh)
                    for t in range(pre,start):
                        _,_,h,vh=self._step(packed,t,h,vh)
                policy_sum=torch.zeros((),device=device)
                entropy_sum=torch.zeros((),device=device)
                kl_sum=torch.zeros((),device=device)
                clip_sum=torch.zeros((),device=device)
                value_sum=torch.zeros((),device=device)
                agent_count=torch.zeros((),device=device)
                time_count=torch.zeros((),device=device)
                norm_entropy_sum=torch.zeros((),device=device)
                amp_context=torch.amp.autocast('cuda') if self.amp else nullcontext()
                with amp_context:
                    for t in range(start,end):
                        dist,v,h,vh=self._step(packed,t,h,vh)
                        act=ex['actions'][:,t].reshape(-1)
                        logp=dist.log_prob(act).reshape(B,N)
                        old=ex['old_logp'][:,t]
                        # The sampled log-prob is only meaningful for due agents.
                        due=ex['decide'][:,t] & obs['active'][:,t] & valid[:,t,None]
                        ratio=torch.exp((logp-old).clamp(-30.,30.))
                        advantage=((ex['advantage'][:,t]-mean)/std)[:,None]
                        if self.cfg.marginal_credit_weight:
                            marginal=(raw_credit[:,t]-cmean)/cstd
                            w=self.cfg.marginal_credit_weight
                            advantage=(1-w)*advantage+w*marginal
                        obj=torch.minimum(ratio*advantage,
                                           ratio.clamp(1-self.cfg.clip,1+self.cfg.clip)*advantage)
                        f=due.float()
                        policy_sum=policy_sum+(obj*f).sum()
                        ent=dist.entropy().reshape(B,N)
                        entropy_sum=entropy_sum+(ent*f).sum()
                        legal_count=obs['legal'][:,t].sum(dim=-1).to(ent.dtype).clamp_min(2)
                        norm_entropy_sum=norm_entropy_sum+((ent/torch.log(legal_count))*f).sum()
                        kl_sum=kl_sum+((old-logp)*f).sum()
                        clip_sum=clip_sum+(((ratio-1).abs()>self.cfg.clip).float()*f).sum()
                        agent_count=agent_count+f.sum()
                        target=ex['return'][:,t]/self.cfg.value_output_scale
                        value_sum=value_sum+((v-target).square()*valid[:,t]).sum()
                        time_count=time_count+valid[:,t].sum()
                    if not bool(agent_count.item()) or not bool(time_count.item()):
                        continue
                    policy=-policy_sum/agent_count
                    entropy=entropy_sum/agent_count
                    vloss=value_sum/time_count
                    loss=policy-self.entropy_coef*entropy+self.cfg.value_coef*vloss
                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.actor_opt)
                self.scaler.unscale_(self.critic_opt)
                ag=nn.utils.clip_grad_norm_(self.actor.parameters(),self.cfg.grad_clip)
                cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.cfg.grad_clip)
                self.scaler.step(self.actor_opt);self.scaler.step(self.critic_opt)
                self.scaler.update()
                metrics.append(dict(policy_loss=float(policy.item()),
                                    value_loss=float(vloss.item()),
                                    entropy=float(entropy.item()),
                                    approx_kl=float((kl_sum/agent_count).item()),
                                    clip_fraction=float((clip_sum/agent_count).item()),
                                    actor_grad_norm=float(ag),critic_grad_norm=float(cg),
                                    actor_decisions=int(agent_count.item()),
                                    decision_boundaries=int(time_count.item()),
                                    normalized_entropy=float((norm_entropy_sum/agent_count).item()),
                                    value_loss_raw_units=float(vloss.item()*self.cfg.value_output_scale**2)))
                entropy_ratios.append(float((norm_entropy_sum/agent_count).item()))
        if not metrics:
            raise RuntimeError('no legal agent decisions collected')
        result={k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}
        result['entropy_coefficient']=self.entropy_coef
        result['marginal_credit_nonzero_fraction']=float((raw_credit[credit_mask]>0).float().mean().item()) if bool(credit_mask.any()) else 0.0
        if self.cfg.adaptive_entropy and entropy_ratios:
            # Entropy adaptation only sees the locally legal categorical
            # distribution, never centralized state or a target assignment.
            import math
            delta=self.cfg.entropy_target_fraction-float(np.mean(entropy_ratios))
            self.entropy_coef=float(np.clip(self.entropy_coef*math.exp(self.cfg.entropy_adapt_rate*delta),
                                            self.cfg.entropy_min,self.cfg.entropy_max))
        return result

    def state_dict(self):
        return dict(actor_opt=self.actor_opt.state_dict(),critic_opt=self.critic_opt.state_dict(),
                    amp_scaler=self.scaler.state_dict(),entropy_coef=self.entropy_coef)

    def load_state_dict(self,data):
        self.actor_opt.load_state_dict(data['actor_opt'])
        self.critic_opt.load_state_dict(data['critic_opt'])
        if 'amp_scaler' in data:self.scaler.load_state_dict(data['amp_scaler'])
        self.entropy_coef=float(data.get('entropy_coef',self.cfg.entropy))
