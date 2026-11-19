import os
import csv
import math
import argparse
from itertools import combinations
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from envs.RoverEnv import RoverEnv
from train import BASE, build_models, rollout, behaviour
from plot import plot_all

# hand-coded baselines, all on the same controller
BASELINES = ("greedy", "switch", "local", "sticky", "fixed")


"""
parse_args
    Parses which policies to evaluate, and on which episodes.

Returns:
    argparse.Namespace of arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rover teams: python evaluate.py checkpoints/*/step500.pth --baselines greedy local fixed")

    # policy args
    parser.add_argument("checkpoints", type=str, nargs="*", help="policy checkpoints (.pth) to evaluate")
    parser.add_argument("--baselines", type=str, nargs="*", default=[], choices=BASELINES,
                        help="greedy: centralized, replans every step; switch: centralized Switch-Aware (exact coalition search, replans on events, keeps rovers on their POI); local: greedy on only the rovers each rover hears; sticky: local with shared, kept commitments; fixed: greedy's first assignment, never replanned")
    parser.add_argument("--comm-radius", type=float, nargs="+", default=[15.0], help="communication radii of the local and sticky baselines (each is evaluated at every one)")
    parser.add_argument("--env-config", type=str, default=f"{BASE}/configs/default.yaml", help="map for the baselines (checkpoints use their own)")

    # episode args
    parser.add_argument("--failures", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="exact failures per episode, each played with random and adversarial picks")
    parser.add_argument("--episodes", type=int, default=64, help="worlds per scenario")
    parser.add_argument("--seed", type=int, default=12345, help="episode seed, the same for every policy (paired)")
    parser.add_argument("--out", type=str, default=f"{BASE}/checkpoints/eval.csv", help="per-checkpoint rows")

    args = parser.parse_args()
    assert args.checkpoints or args.baselines, "give checkpoints and/or --baselines"
    return args


"""
GreedyAllocator
    Hand-coded baseline with the same controller. Greedy
    allocation: available POIs in order of value each claim
    their `coupling` nearest unclaimed rovers (skipped if too
    few are left), and leftover rovers idle. Four variants:
        greedy: centralized, one allocation over every living
            rover, replanned every step (so it reallocates
            immediately after a failure);
        local: each rover runs the allocation over only itself
            and the rovers it hears, and follows its own part of
            it (the same information as the learned policy);
        sticky: local, with commitment. Rovers share their
            current POI with the rovers they hear, and keep it
            while it is available and they are not surplus to it
            (among the `coupling` nearest of the rovers they know
            are committed to it). An uncommitted rover plans over
            the uncommitted rovers it hears, for what the
            committed ones it hears don't already cover, and
            claims a POI even when its view has too few rovers
            for it (others may come);
        fixed: the centralized allocation of the first step,
            never replanned (a rover idles once its POI closes).
"""
class GreedyAllocator:
    """
    GreedyAllocator.__init__
        Args:
            env: RoverEnv it reads the true state from
            mode: string 'greedy', 'local', 'sticky' or 'fixed'
    """
    def __init__(self, env, mode="greedy"):
        self.env = env
        self.mode = mode
        self.plan = None

    """
    GreedyAllocator.allocate
        Greedy allocation, run separately for every rover's view
        of the team (all views share the POIs).

        Args:
            free: torch.Tensor of size (B, A, A) bool, rover j can be assigned in rover i's view
            available: torch.Tensor of size (B, P) bool POIs that can still pay
            need: torch.Tensor of size (B, A, P) rovers each POI still needs in each view
                (default: its coupling)
            optimistic: bool claim a POI even if the view has fewer free rovers than it needs

        Returns:
            torch.Tensor of size (B, A) each rover's own POI in its view, P = idle
    """
    def allocate(self, free, available, need=None, optimistic=False):
        env = self.env
        B, A, P = env.B, env.A, env.P
        if need is None:
            need = env.coupling[:, None, :].expand(B, A, P)
        dist = torch.cdist(env.pos, env.poi_pos)
        order = env.value.argsort(-1, descending=True)
        action = torch.full((B, A), P, device=env.device)
        free = free.clone()
        for rank in range(P):
            p = order[:, rank]
            n = need.gather(2, p[:, None, None].expand(B, A, 1))
            ok = available.gather(1, p[:, None])[:, :, None] & (n > 0)
            if not optimistic:
                ok &= free.sum(-1, keepdim=True) >= n

            # in every view, the `need` nearest free rovers to p claim it
            d = dist.gather(2, p[:, None, None].expand(B, A, 1)).squeeze(-1)
            d = d[:, None, :].expand(B, A, A).masked_fill(~free, math.inf)
            chosen = free & ok & (d.argsort(-1).argsort(-1) < n)
            mine = chosen.diagonal(dim1=1, dim2=2)
            action = torch.where(mine, p[:, None], action)
            free &= ~chosen
        return action

    """
    GreedyAllocator.sticky
        One step of sticky local greedy (see the class).

        Args:
            known: torch.Tensor of size (B, A, A) bool, rover i hears rover j (or j == i)
            available: torch.Tensor of size (B, P) bool POIs that can still pay

        Returns:
            torch.Tensor of size (B, A) each rover's POI, P = idle
    """
    def sticky(self, known, available):
        env = self.env
        B, A, P = env.B, env.A, env.P
        if self.plan is None or env.t == 0:
            self.plan = torch.full((B, A), P, device=env.device)
        plan, poi = self.plan, self.plan.clamp(max=P - 1)
        keep = known.diagonal(dim1=1, dim2=2) & (plan < P) & available.gather(1, poi)

        # surplus: rover i ranks itself among the rovers it knows keep the same POI, by distance to it
        same = known & keep[:, None, :] & (plan[:, None, :] == plan[:, :, None])
        d = torch.cdist(env.pos, env.poi_pos).gather(2, poi[:, None, :].expand(B, A, A)).transpose(1, 2)
        own = d.diagonal(dim1=1, dim2=2)[:, :, None]
        index = torch.arange(A, device=env.device)
        ahead = same & ((d < own) | ((d == own) & (index[None, None, :] < index[None, :, None])))
        committed = keep & (ahead.sum(-1) < env.coupling.gather(1, poi))

        # uncommitted rovers plan for what the committed rovers they hear don't cover
        covered = (known & committed[:, None, :])[..., None] & (F.one_hot(poi, P).bool() & (plan < P)[..., None])[:, None]
        need = (env.coupling[:, None, :] - covered.sum(2)).clamp(min=0)
        choice = self.allocate(known & ~committed[:, None, :], available, need, optimistic=True)
        self.plan = torch.where(committed, plan, choice)
        return self.plan

    """
    GreedyAllocator.__call__
        Args:
            obs: dict of observation tensors (uses alive, comm; POI status is read from the env)

        Returns:
            torch.distributions.Categorical with all mass on the allocation
    """
    def __call__(self, obs):
        env = self.env
        available = env.available()
        alive = obs['alive'][:, None, :].expand(env.B, env.A, env.A)
        eye = torch.eye(env.A, dtype=torch.bool, device=env.device)
        if self.mode == "local":
            action = self.allocate(alive & (obs['comm'] | eye), available)
        elif self.mode == "sticky":
            action = self.sticky(alive & (obs['comm'] | eye), available)
        elif self.mode == "fixed":
            if env.t == 0:
                self.plan = self.allocate(alive, available)
            open_ = available.gather(1, self.plan.clamp(max=env.P - 1)) | (self.plan == env.P)
            action = torch.where(open_, self.plan, env.P)
        else:
            action = self.allocate(alive, available)
        logits = torch.full((env.B, env.A, env.P + 1), -1e9, device=env.device)
        return Categorical(logits=logits.scatter(-1, action[..., None], 0.0))


"""
SwitchAware
    Centralized Switch-Aware allocator from DART-MARL (ported
    from its roles/allocator.py). It plans only on events: the
    first step, a failure, or a POI opening, closing or
    completing; between events every rover keeps its POI. A plan
    is the exact best set of disjoint, fully staffed coalitions
    (exactly `coupling` living rovers per POI, able to arrive
    before the window closes), scored by
        remaining value - travel_weight * travel
            + switch_weight * (members already on that POI),
    where remaining value is the POI's value times the steps
    from the coalition's arrival to the window's close (once
    for one-shot POIs). Found by dynamic programming over
    subsets of rovers. Unassigned rovers idle.
"""
class SwitchAware:
    """
    SwitchAware.__init__
        Args:
            env: RoverEnv it reads the true state from
            travel_weight: float cost per step of summed travel time
            switch_weight: float credit per rover kept on its POI
    """
    def __init__(self, env, travel_weight=0.1, switch_weight=3.0):
        self.env = env
        self.travel_weight = travel_weight
        self.switch_weight = switch_weight
        self.masks = np.arange(1 << env.A)
        self.plan = self.status = None

    """
    SwitchAware.allocate
        Exact coalition allocation for one world.

        Args:
            b: int world index
            previous: np.ndarray of size (A,) each rover's current POI, P = idle

        Returns:
            np.ndarray of size (A,) each rover's POI, P = idle
    """
    def allocate(self, b, previous):
        env = self.env
        t, A, P = env.t, env.A, env.P
        alive = np.flatnonzero(env.active[b].cpu().numpy())
        pos, poi_pos = env.pos[b].cpu().numpy(), env.poi_pos[b].cpu().numpy()
        radius, window = env.radius[b].cpu().numpy(), env.window[b].cpu().numpy()
        value, repeat, coupling = env.value[b].cpu().numpy(), env.repeat[b].cpu().numpy(), env.coupling[b].cpu().numpy()
        available = env.available()[b].cpu().numpy()

        # dp[mask]: best score using exactly the rovers in mask; one (choice, parent) table per POI for backtracking
        dp = np.full(1 << A, -np.inf)
        dp[0] = 0.0
        tables = []
        for p in range(P):
            choice, parent, new = np.full(1 << A, -1), self.masks.copy(), dp.copy()
            coalitions = []
            if available[p] and len(alive) >= coupling[p]:
                travel = np.maximum(np.linalg.norm(pos - poi_pos[p], axis=-1) - radius[p], 0.0) / env.speed
                for members in combinations(alive.tolist(), int(coupling[p])):
                    start = max(int(window[p, 0]), t + int(np.ceil(travel[list(members)].max() - 1e-9)))
                    if start > window[p, 1]:
                        continue
                    gain = value[p] * (window[p, 1] - start + 1 if repeat[p] else 1)
                    score = (gain - self.travel_weight * travel[list(members)].sum()
                             + self.switch_weight * sum(previous[i] == p for i in members))
                    coalitions.append((members, sum(1 << i for i in members), score))
            for c, (members, m, score) in enumerate(coalitions):
                ok = ((self.masks & m) == 0) & np.isfinite(dp)
                source = self.masks[ok]
                target, cand = source | m, dp[ok] + score
                better = cand > new[target]
                new[target[better]] = cand[better]
                choice[target[better]], parent[target[better]] = c, source[better]
            tables.append((coalitions, choice, parent))
            dp = new

        # backtrack from the best final set of rovers
        action, mask = np.full(A, P), int(np.argmax(dp))
        for p in reversed(range(P)):
            coalitions, choice, parent = tables[p]
            if choice[mask] >= 0:
                action[list(coalitions[choice[mask]][0])] = p
                mask = int(parent[mask])
        return action

    """
    SwitchAware.__call__
        Replans the worlds with an event this step, and keeps
        every other world's plan.

        Args:
            obs: dict of observation tensors (unused: the allocator reads the env)

        Returns:
            torch.distributions.Categorical with all mass on the allocation
    """
    def __call__(self, obs):
        env = self.env
        opened = env.window[..., 0] <= env.t
        status = torch.cat([env.active, env.available(), opened], dim=-1).cpu().numpy()
        if env.t == 0:
            self.plan = np.full((env.B, env.A), env.P)
        for b in range(env.B):
            if env.t == 0 or (status[b] != self.status[b]).any():
                self.plan[b] = self.allocate(b, self.plan[b])
        self.status = status
        action = torch.as_tensor(self.plan, device=env.device)
        logits = torch.full((env.B, env.A, env.P + 1), -1e9, device=env.device)
        return Categorical(logits=logits.scatter(-1, action[..., None], 0.0))


"""
km
    Kaplan-Meier estimate for right-censored times: the
    probability the event has happened by each time in `at`, and
    the median time (inf if fewer than half are ever observed).

Args:
    times: list of float times (event, or censoring)
    observed: list of bool, True if the event happened at that time
    at: list of float times to evaluate

Returns:
    list of float probabilities, one per `at`
    float median time
"""
def km(times, observed, at):
    survival, median, curve = 1.0, math.inf, []
    for u in sorted({t for t, o in zip(times, observed) if o}):
        at_risk = sum(t >= u for t in times)
        events = sum(t == u and o for t, o in zip(times, observed))
        survival *= 1 - events / at_risk
        curve.append((u, survival))
        if survival <= 0.5 and median == math.inf:
            median = u
    done = [1 - min([s for u, s in curve if u <= a], default=1.0) for a in at]
    return done, median


"""
recovery
    Restoration of POI service after failures. A failure puts a
    POI at risk when the failed rover was essential to it (its
    difference reward was positive on the step before). The lag
    is the number of steps until the POI is served again, and the
    sustained lag until it is served on 2 consecutive steps. Both
    are right-censored at the world's next failure, the POI's
    window closing, or the episode's end. The minimum lag is the
    travel time of the nearest free rover (alive, not essential);
    the excess lag (lag - minimum) is the time spent noticing and
    deciding. `heard` marks failures with a free rover in comm
    range of the failed rover.

Args:
    batch: dict from rollout
    env: RoverEnv the batch was played in

Returns:
    list of dict events: lag, sustained, excess (each with an
        _observed flag), heard
"""
def recovery(batch, env):
    alive, served, difference = batch['obs']['alive'], batch['served'], batch['difference']
    pos, comm = batch['obs']['pos'] * env.dims, batch['obs']['comm']
    T = alive.shape[0]
    failed = (alive[:-1] & ~alive[1:]).nonzero().tolist()
    fail_steps = defaultdict(list)
    for s, b, i in failed:
        fail_steps[b].append(s + 1)

    events = []
    for s, b, i in failed:
        s += 1
        if difference[s - 1, b, i] <= 0:
            continue

        # the POI it covered (the positions in obs[s] are where coverage was counted on step s - 1)
        d = (pos[s, b, i] - env.poi_pos[b]).norm(dim=-1)
        inside = served[s - 1, b] & (d < env.radius[b]) & env.repeat[b]
        if not inside.any():
            continue
        p = torch.where(inside, d / env.radius[b], math.inf).argmin().item()

        # follow the POI until the next failure in this world, its window closing, or the episode's end
        end = min([t for t in fail_steps[b] if t > s] + [int(env.window[b, p, 1]) + 1, T])
        ok = served[s:end, b, p]
        both = ok[:-1] & ok[1:]
        lag = ok.nonzero()[0].item() if ok.any() else end - s
        sustained = both.nonzero()[0].item() if both.any() else end - s

        # fastest possible restoration: the nearest free rover's travel time
        free = alive[s, b] & (difference[s - 1, b] <= 0)
        dj = (pos[s, b] - env.poi_pos[b, p]).norm(dim=-1)
        moves = torch.where(dj < env.radius[b, p], 0.0, (dj - env.radius[b, p]).floor() + 1)
        tmin = (moves - 1).clamp(min=0).masked_fill(~free, math.inf).min().item()
        events.append({
            'lag': lag, 'lag_observed': ok.any().item(),
            'sustained': sustained, 'sustained_observed': both.any().item(),
            'excess': lag - tmin if tmin < math.inf else math.inf, 'excess_observed': ok.any().item() and tmin < math.inf,
            'heard': (comm[s - 1, b, i] & free).any().item()})
    return events


"""
summarize
    Recovery statistics from recovery events (see km).

Args:
    events: list of dict from recovery
    episodes: int episodes played

Returns:
    dict of at_risk (per episode), recovered_10, recovered_20,
        lag_median, sustained_median, excess_median,
        excess_heard_median, excess_unheard_median
"""
def summarize(events, episodes):
    def median(key, subset):
        subset = [e for e in subset if key != 'excess' or e['excess'] < math.inf]
        if not subset:
            return math.nan
        return km([e[key] for e in subset], [e[key + '_observed'] for e in subset], [])[1]
    stats = {'at_risk': len(events) / episodes}
    if events:
        (r10, r20), _ = km([e['lag'] for e in events], [e['lag_observed'] for e in events], [10, 20])
        stats.update(recovered_10=r10, recovered_20=r20)
    else:
        stats.update(recovered_10=math.nan, recovered_20=math.nan)
    stats.update(
        lag_median=median('lag', events),
        sustained_median=median('sustained', events),
        excess_median=median('excess', events),
        excess_heard_median=median('excess', [e for e in events if e['heard']]),
        excess_unheard_median=median('excess', [e for e in events if not e['heard']]))
    return stats


"""
recovery_curves
    Recovery over time, for plotting. 'restored': the share of
    at-risk POIs served again within s steps of the failure
    (Kaplan-Meier, censored as in recovery). With one failure
    only, aligned at the failure step and averaged over episodes:
    'reward', the team reward per step; and 'reward_lost', the
    reward lost per step relative to the same episode without the
    failure (0 before the failure; its return to 0 is the
    recovery). Each point also has the standard error of its mean
    over episodes ('sem').

Args:
    row: dict identifying the scenario (name, seed, update, failures, adversary)
    events: list of dict from recovery
    batch: dict from rollout
    nominal_reward: torch.Tensor of size (T, B) rewards of the episodes without failures
    before: int steps before the failure
    after: int steps after the failure

Returns:
    list of dict rows: the row's keys, curve, step, value
"""
def recovery_curves(row, events, batch, nominal_reward, before=10, after=50):
    curves = []
    if events:
        restored, _ = km([e['lag'] for e in events], [e['lag_observed'] for e in events], list(range(after + 1)))
        curves += [{**row, 'curve': 'restored', 'step': s, 'value': v} for s, v in enumerate(restored)]
    if row['failures'] == 1:
        alive, T = batch['obs']['alive'], batch['reward'].shape[0]
        failed = (alive[:-1] & ~alive[1:]).any(-1)
        for curve, values in [('reward', batch['reward']), ('reward_lost', nominal_reward - batch['reward'])]:
            trace = torch.full((before + after + 1, values.shape[1]), math.nan)
            for s, b in failed.nonzero().tolist():
                lo, hi = max(s + 1 - before, 0), min(s + 1 + after + 1, T)
                trace[lo - (s + 1 - before):hi - (s + 1 - before), b] = values[lo:hi, b]
            # mean and standard error over the episodes that cover each step
            n = (~trace.isnan()).sum(1)
            avg = trace.nanmean(1)
            sem = (((trace - avg[:, None]) ** 2).nansum(1) / (n - 1).clamp(min=1) / n.clamp(min=1)).sqrt()
            curves += [{**row, 'curve': curve, 'step': o - before, 'value': v, 'sem': e}
                       for o, (v, e) in enumerate(zip(avg.tolist(), sem.tolist()))]
    return curves


"""
evaluate
    Plays a policy deterministically on the same episodes (same
    seed, so the same rover starts) with each exact number of
    failures, picked at random and by the greedy adversary.
    Random picks are paired across policies; adversarial picks
    depend on what the policy does. The fault cost of an episode
    is its return without failures minus its return with them
    (the same episode otherwise).

Args:
    path: string path to a .pth checkpoint, or a baseline name
    episodes: int worlds per scenario
    seed: int episode seed
    env_config: string yaml map (baselines only; checkpoints use their own)
    comm_radius: float comm radius (baselines only; checkpoints use their own)
    failures: list of int failures per episode (0 is played once)

Returns:
    list of dict rows: name, seed, update, failures, adversary, return,
        fault_cost, cost_per_failure, coverage, recovery stats, switch_rate, idle_rate
    list of dict recovery curve rows (see recovery_curves)
"""
def evaluate(path, episodes, seed, env_config, comm_radius, failures):
    # the checkpoint's arguments and policy weights (none for baselines)
    if path in BASELINES:
        name = {'greedy': 'greedy', 'switch': 'switch_aware', 'local': f'local_r{comm_radius:g}', 'sticky': f'sticky_r{comm_radius:g}', 'fixed': 'fixed'}[path]
        args, update, state = {'name': name, 'seed': None, 'env_config': env_config, 'comm_radius': comm_radius}, None, None
    else:
        chkpt = torch.load(path, map_location="cpu")
        args, update, state = chkpt['args'], chkpt['update'], chkpt['policy_state_dict']

    def play(k, adversary):
        env = RoverEnv(args['env_config'], episodes, args['comm_radius'], seed=seed)
        if state is None:
            policy = SwitchAware(env) if path == "switch" else GreedyAllocator(env, path)
        else:
            policy, _ = build_models(args, env)
            policy.load_state_dict(state)
        faults = {'min_faults': k, 'max_faults': k, 'adversarial': float(adversary == "adversarial")}
        return env, rollout(env, policy, faults, deterministic=True)

    # the failure-free episodes every scenario is compared against
    _, nominal = play(0, "none")
    nominal_return = nominal['reward'].sum(0)

    rows, curves = [], []
    scenarios = [(k, adversary) for k in failures for adversary in (["none"] if k == 0 else ["random", "adversarial"])]
    for k, adversary in scenarios:
        env, batch = play(k, adversary)
        assert torch.equal(batch['obs']['pos'][0], nominal['obs']['pos'][0]), "episodes are not paired"
        ret = batch['reward'].sum(0)
        cost = (nominal_return - ret).mean().item()
        events = recovery(batch, env)
        key = {'name': args['name'], 'seed': args['seed'], 'update': update, 'comm_radius': args['comm_radius'], 'failures': k, 'adversary': adversary}
        curves += recovery_curves(key, events, batch, nominal['reward'])
        rows.append({
            **key,
            'return': ret.mean().item(),
            'return_sem': (ret.std() / math.sqrt(len(ret))).item(),
            'fault_cost': cost,
            'cost_per_failure': cost / k if k else 0.0,
            'coverage': batch['served'].float().mean().item(),
            **summarize(events, episodes),
            **behaviour(batch, env.P)})
    return rows, curves


"""
report
    Writes every row to a csv, and prints means over the
    checkpoints of each run name (e.g. one per seed) per number
    of failures and adversary: return (± std over checkpoints),
    fault cost per failure, at-risk POIs per episode, the share
    restored within 10 steps, the median lag and median excess
    lag (Kaplan-Meier, censored), and the switch rate.

    Recovery curves go to a second csv next to it
    (<out>_recovery.csv).

Args:
    rows: list of dict rows from evaluate
    curves: list of dict recovery curve rows from evaluate
    out: string csv path
"""
def report(rows, curves, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for path, table in [(out, rows), (out.replace('.csv', '_recovery.csv'), curves)]:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, list(dict.fromkeys(k for row in table for k in row)))   # every column any row has
            writer.writeheader()
            writer.writerows(table)

    # group by run name, failures and adversary
    groups = defaultdict(list)
    for row in rows:
        groups[row['name'], row['failures'], row['adversary']].append(row)
    print(f"{'name':14}{'fail':>5}{'adversary':>12}{'return':>16}{'cost/f':>8}{'at-risk':>8}{'rec@10':>8}"
          f"{'lag':>6}{'excess':>7}{'heard':>6}{'unheard':>8}{'switch':>8}  n")
    for (name, k, adversary), group in groups.items():
        ret = torch.tensor([r['return'] for r in group])
        std = ret.std().item() if len(group) > 1 else 0.0
        mean = lambda key: sum(r[key] for r in group) / len(group)
        fmt = lambda x: "-" if math.isnan(x) else ("inf" if x == math.inf else f"{x:.0f}")
        recovered = "-" if math.isnan(mean('recovered_10')) else f"{mean('recovered_10'):.0%}"
        print(f"{name:14}{k:>5}{adversary:>12}{ret.mean():>9.1f} ± {std:<4.1f}{mean('cost_per_failure'):>8.1f}"
              f"{mean('at_risk'):>8.2f}{recovered:>8}{fmt(mean('lag_median')):>6}{fmt(mean('excess_median')):>7}"
              f"{fmt(mean('excess_heard_median')):>6}{fmt(mean('excess_unheard_median')):>8}{mean('switch_rate'):>8.1%}  {len(group)}")
    print(f"per-checkpoint rows -> {out} (+ _recovery.csv)")


"""
main
    Evaluates the baselines and every checkpoint on the same
    episodes, reports, and draws the figures (see plot.plot_all)
    next to the csv.

Args:
    args: argparse.Namespace of arguments
"""
def main(args):
    rows, curves = [], []
    # centralized baselines once, decentralized ones at every radius, checkpoints at their own radius
    runs = [(b, radius) for b in args.baselines for radius in (args.comm_radius if b in ("local", "sticky") else [math.inf])]
    for path, radius in runs + [(c, None) for c in args.checkpoints]:
        r, c = evaluate(path, args.episodes, args.seed, args.env_config, radius, args.failures)
        rows, curves = rows + r, curves + c
    report(rows, curves, args.out)

    # figures next to the csv: failure sweep, reward around one failure, radius sweep (local baselines at r = 15 if evaluated)
    radius = 15.0 if 15.0 in args.comm_radius else args.comm_radius[0]
    plot_all(args.out, os.path.dirname(os.path.abspath(args.out)), baseline_radius=radius)


if __name__ == "__main__":
    main(parse_args())
