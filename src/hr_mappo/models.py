"""Task-conditioned categorical shared actor and training-only team-value critic."""
import torch
from torch import nn
from torch.distributions import Categorical
from .observations import EGO, TASK, NEIGHBOR, STATUS


def mlp(a,b,h):
    return nn.Sequential(nn.Linear(a,h), nn.Tanh(), nn.Linear(h,b), nn.Tanh())

class TaskActor(nn.Module):
    def __init__(self, hidden=128, recurrent=True):
        super().__init__()
        self.hidden=hidden
        self.recurrent=recurrent
        self.task=mlp(TASK,hidden,hidden)
        self.ego=mlp(EGO,hidden,hidden)
        self.neighbor=mlp(NEIGHBOR,hidden,hidden)
        self.status=mlp(STATUS,hidden,hidden)
        self.combine=mlp(hidden*4,hidden,hidden)
        self.gru=nn.GRUCell(hidden,hidden) if recurrent else None
        self.key=nn.Linear(hidden,hidden)
        self.query=nn.Linear(hidden,hidden)
        self.idle=nn.Linear(hidden,1)

    def forward_step(self, obs, h):
        # All tensors have first axis N; caller controls per-agent lifecycle.
        task=self.task(obs['tasks'])
        neighbor=self.neighbor(obs['neighbors'])
        m=obs['neighbor_mask'].unsqueeze(-1).to(neighbor.dtype)
        pooled=(neighbor*m).sum(1)/m.sum(1).clamp_min(1.)
        tm=obs['legal'][:,:-1].unsqueeze(-1).to(task.dtype)
        tp=(task*tm).sum(1)/tm.sum(1).clamp_min(1.)
        sm=obs['status_mask'].unsqueeze(-1).to(task.dtype)
        encoded=self.status(obs['status'])
        sp=(encoded*sm).sum(1)/sm.sum(1).clamp_min(1.)
        x=self.combine(torch.cat([self.ego(obs['ego']),tp,pooled,sp],-1))
        candidate=self.gru(x,h) if self.recurrent else x
        # An inactive slot does not advance its memory; lifecycle reset is
        # explicit in the collector and exactly replayed in PPO.
        h=torch.where(obs['active'].unsqueeze(-1),candidate,h)
        logits=(self.key(task)*self.query(h).unsqueeze(1)).sum(-1)/(self.hidden**.5)
        logits=torch.cat([logits,self.idle(h)],-1)
        logits=logits.masked_fill(~obs['legal'], -1e9)
        return logits,h

    def distribution(self,obs,h):
        logits,h=self.forward_step(obs,h)
        return Categorical(logits=logits),h

class TeamCritic(nn.Module):
    def __init__(self,hidden=128,recurrent=True):
        super().__init__()
        self.hidden=hidden
        self.recurrent=recurrent
        self.agent=mlp(7,hidden,hidden)
        self.poi=mlp(9,hidden,hidden)
        self.combine=mlp(hidden*2+1,hidden,hidden)
        self.gru=nn.GRUCell(hidden,hidden) if recurrent else None
        self.out=nn.Linear(hidden,1)

    def forward_step(self, state,h):
        a=self.agent(state['agent']); p=self.poi(state['poi'])
        am=state['active'].unsqueeze(-1).to(a.dtype)
        pm=state['poi_mask'].unsqueeze(-1).to(p.dtype)
        a=(a*am).sum(-2)/am.sum(-2).clamp_min(1)
        p=(p*pm).sum(-2)/pm.sum(-2).clamp_min(1)
        x=self.combine(torch.cat([a,p,state['clock']],-1))
        if x.ndim == 1: x=x.unsqueeze(0)
        h=self.gru(x,h) if self.recurrent else x
        return self.out(h).squeeze(-1),h
