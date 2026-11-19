import argparse

import torch
import rerun as rr
import rerun.blueprint as rrb

from envs.RoverEnv import RoverEnv
from train import BASE, build_models, rollout
from evaluate import BASELINES, GreedyAllocator

# one colour per POI (rovers take the colour of the POI they chose), then idle and failed rovers
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 200, 200)]
IDLE, FAILED, LINK = (170, 170, 170), (70, 70, 70), (120, 120, 120, 60)


"""
parse_args
    Parses which policy to play, and on which episode.

Returns:
    argparse.Namespace of arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Record one episode for the Rerun viewer: python visualize.py checkpoints/<run>/step500.pth --failures 3")

    # policy args
    parser.add_argument("policy", type=str, help=f"policy checkpoint (.pth), or a baseline: {', '.join(BASELINES)}")
    parser.add_argument("--comm-radius", type=float, default=15.0, help="communication radius of a baseline (checkpoints use their own)")
    parser.add_argument("--env-config", type=str, default=f"{BASE}/configs/rover_team_12.yaml", help="map for a baseline (checkpoints use their own)")

    # episode args
    parser.add_argument("--failures", type=int, default=0, help="exact failures in the episode")
    parser.add_argument("--adversarial", action="store_true", help="failures picked by the greedy adversary (default: at random)")
    parser.add_argument("--seed", type=int, default=0, help="episode seed: rover starts and failure steps")
    parser.add_argument("--out", type=str, default=f"{BASE}/checkpoints/episode.rrd", help="recording to open with `rerun <file>`")
    return parser.parse_args()


"""
load
    Builds a one-world environment and the policy to play in it.

Args:
    args: argparse.Namespace of arguments

Returns:
    RoverEnv
    policy (GraphTransformer or GreedyAllocator)
    string name
"""
def load(args):
    if args.policy in BASELINES:
        env = RoverEnv(args.env_config, 1, args.comm_radius, seed=args.seed)
        return env, GreedyAllocator(env, args.policy), f"{args.policy}_r{args.comm_radius:g}"
    chkpt = torch.load(args.policy, map_location="cpu")
    env = RoverEnv(chkpt['args']['env_config'], 1, chkpt['args']['comm_radius'], seed=args.seed)
    policy, _ = build_models(chkpt['args'], env)
    policy.load_state_dict(chkpt['policy_state_dict'])
    return env, policy, f"{chkpt['args']['name']}_step{chkpt['update']}"


"""
log_episode
    Logs every step of a one-world episode to Rerun: the map
    (POIs filled when served, sized by their radius), rovers
    coloured by the POI they chose, lines to their chosen POIs,
    comm links, failures as events, and the team reward, return
    and living rovers over time.

Args:
    env: RoverEnv the episode was played in
    batch: dict from rollout
"""
def log_episode(env, batch):
    dims, P = env.dims, env.P
    flip = lambda xy: torch.stack([xy[..., 0], dims[1] - xy[..., 1]], dim=-1)   # Rerun's 2D y axis points down
    poi_xy = flip(env.poi_pos[0])
    rr.log("map/bounds", rr.Boxes2D(mins=[0, 0], sizes=dims.tolist(), colors=[(200, 200, 200)]), static=True)

    obs, ret = batch['obs'], 0.0
    for t in range(env.horizon):
        rr.set_time("step", sequence=t)
        pos = flip(obs['pos'][t, 0] * dims)
        alive, action, served = obs['alive'][t, 0], batch['action'][t, 0], batch['served'][t, 0]

        # POIs: filled while served, faint otherwise, labelled with how many rovers they need
        rr.log("map/pois", rr.Points2D(
            poi_xy, radii=env.radius[0],
            colors=[COLORS[p] + ((200,) if served[p] else (50,)) for p in range(P)],
            labels=[f"P{p} x{int(env.coupling[0, p])}" + ("" if env.repeat[0, p] else " once") for p in range(P)],
            draw_order=0))

        # rovers coloured by their chosen POI, with a line to it; failed rovers dark
        colors = [FAILED if not alive[i] else IDLE if action[i] == P else COLORS[action[i]] for i in range(env.A)]
        rr.log("map/rovers", rr.Points2D(pos, radii=0.6, colors=colors, labels=[str(i) for i in range(env.A)], draw_order=3))
        going = [i for i in range(env.A) if alive[i] and action[i] < P]
        rr.log("map/allocation", rr.LineStrips2D(
            [[pos[i].tolist(), poi_xy[action[i]].tolist()] for i in going], colors=[COLORS[action[i]] for i in going], draw_order=2))

        # who hears whom
        links = obs['comm'][t, 0].triu(1).nonzero().tolist()
        rr.log("map/comm", rr.LineStrips2D([[pos[i].tolist(), pos[j].tolist()] for i, j in links], colors=[LINK], draw_order=1))

        # failures, and team stats
        if t > 0:
            for i in (obs['alive'][t - 1, 0] & ~alive).nonzero().flatten().tolist():
                rr.log("events", rr.TextLog(f"rover {i} failed", level=rr.TextLogLevel.WARN))
        ret += batch['reward'][t, 0].item()
        rr.log("team/return", rr.Scalars(ret))
        rr.log("team/reward", rr.Scalars(batch['reward'][t, 0].item()))
        rr.log("team/alive", rr.Scalars(alive.sum().item()))


"""
main
    Plays one episode deterministically and saves it as a Rerun
    recording, with a layout of the map, the team stats and the
    failure log.

Args:
    args: argparse.Namespace of arguments
"""
def main(args):
    env, policy, name = load(args)
    faults = {'min_faults': args.failures, 'max_faults': args.failures, 'adversarial': float(args.adversarial)}
    batch = rollout(env, policy, faults, deterministic=True)

    rr.init(f"marl_project_{name}")
    rr.save(args.out)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial2DView(origin="map", name="map"),
        rrb.Vertical(rrb.TimeSeriesView(origin="team", name="team"), rrb.TextLogView(origin="events", name="failures")),
        column_shares=[3, 2])))
    log_episode(env, batch)
    print(f"{name}: return {batch['reward'].sum().item():.1f} -> {args.out}  (open with: rerun {args.out})")


if __name__ == "__main__":
    main(parse_args())
