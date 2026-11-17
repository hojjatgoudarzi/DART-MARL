import os
import sys
import time
import signal
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from envs.RoverEnv import RoverEnv
from models.GraphTransformer import GraphTransformer
from models.ValueNetwork import ValueNetwork

BASE = os.path.dirname(os.path.abspath(__file__))

# dynamically select device (the default models are small: CPU is often as fast)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
parse_args
    Parses environment, training, and model arguments.

Returns:
    argparse.Namespace of arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Train a MAPPO rover team.")

    # environment args
    parser.add_argument("--env-config", type=str, default=f"{BASE}/configs/rover_team_12.yaml")
    parser.add_argument("--num-envs", type=int, default=128, help="worlds per update, each plays one full episode")
    parser.add_argument("--comm-radius", type=float, default=15.0, help="communication radius in map units (inf = everyone, 0 = no communication)")
    parser.add_argument("--reward", type=str, default="difference", choices=["difference", "team"], help="each rover learns from its difference reward D_i = G - G_-i, or from the shared team reward G")

    # failure curriculum args: warmup without failures; then the updates left are split evenly between
    # a failure ramp (max failures 0 -> --max-faults) and an adversarial ramp (probability 0 -> --adversarial)
    parser.add_argument("--warmup", type=int, default=100, help="updates without failures at the start")
    parser.add_argument("--max-faults", type=int, default=4, help="failures per episode are drawn uniformly from [0, max]; max ramps up to this")
    parser.add_argument("--adversarial", type=float, default=0.5, help="adversarial probability per failure at the last update")

    # training args
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=5e-4, help="learning rate of the value network")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4, help="minibatches of whole episodes per epoch")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-scale", type=float, default=256.0, help="the value network predicts return / value-scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None, help="resume a run: weights, optimizer and update")
    parser.add_argument("--checkpoint-dir", type=str, default=f"{BASE}/checkpoints")
    parser.add_argument("--name", type=str, default=None, help="run name prefix (default: r<comm radius>_<reward>); also a wandb config key to group seeds by")

    # model args
    parser.add_argument("--hidden", type=int, default=64, help="token width of the policy and the value network")
    parser.add_argument("--msg-dim", type=int, default=64, help="broadcast message size (bandwidth)")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2, help="cross-attention layers of the policy (and transformer layers of the value network)")

    args = parser.parse_args()
    if args.name is None:
        args.name = f"r{args.comm_radius:g}_{args.reward}"
    return args


"""
build_models
    Constructs the policy and the value network described by the
    arguments.

Args:
    args: argparse.Namespace (or dict) of arguments
    env: RoverEnv

Returns:
    GraphTransformer policy
    ValueNetwork
"""
def build_models(args, env):
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    policy = GraphTransformer(env.agent_dim, env.poi_dim, args['hidden'], args['msg_dim'], args['heads'], args['layers'])
    value_net = ValueNetwork(env.agent_dim, env.poi_dim, args['hidden'], args['heads'], args['layers'])
    return policy, value_net


"""
curriculum
    Failure settings for one update, like a learning-rate
    schedule. Failures per episode are uniform in [0, max], and
    each is adversarial with some probability (else a uniform
    pick). The updates after warmup are split evenly:
        1. warmup: no failures;
        2. failure ramp: max rises linearly (rounded) from 0 to
           --max-faults, with uniform picks only;
        3. adversarial ramp: max stays at --max-faults, and the
           adversarial probability rises linearly from 0 to
           --adversarial at the last update.

Args:
    args: argparse.Namespace of arguments
    update: int current update (1-indexed)

Returns:
    dict of RoverEnv.reset arguments: min_faults, max_faults, adversarial
"""
def curriculum(args, update):
    ramp = (args.updates - args.warmup) // 2
    if update <= args.warmup:
        max_faults, adversarial = 0, 0.0
    elif update <= args.warmup + ramp:
        max_faults, adversarial = round(args.max_faults * (update - args.warmup) / ramp), 0.0
    else:
        progress = (update - args.warmup - ramp) / (args.updates - args.warmup - ramp)
        max_faults, adversarial = args.max_faults, args.adversarial * min(progress, 1.0)
    return {'min_faults': 0, 'max_faults': max_faults, 'adversarial': adversarial}


"""
dry_run
    Runs both models on real observations, with one and with two
    leading dimensions, to ensure proper dimensionality. Asserts
    correct output shapes.

Args:
    policy: GraphTransformer
    value_net: ValueNetwork
    env: RoverEnv
"""
def dry_run(policy, value_net, env):
    obs = env.reset()
    seq = {k: torch.stack([v, v]) for k, v in obs.items()}
    with torch.no_grad():
        assert policy(obs).logits.shape == (env.B, env.A, env.P + 1)
        assert value_net(obs).shape == (env.B, env.A)
        assert policy(seq).logits.shape == (2, env.B, env.A, env.P + 1)
        assert value_net(seq).shape == (2, env.B, env.A)
    print("[dry_run] passed")


"""
rollout
    Plays one synchronous episode in every world.

Args:
    env: RoverEnv
    policy: GraphTransformer (or anything mapping observations to a Categorical)
    faults: dict of failure arguments of RoverEnv.reset
    deterministic: bool pick the most likely action instead of sampling

Returns:
    dict of torch.Tensor of size (T, B, ...), observations under 'obs'
"""
@torch.no_grad()
def rollout(env, policy, faults, deterministic=False):
    obs = env.reset(**faults)
    steps = []
    for t in range(env.horizon):
        dist = policy(obs)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        next_obs, reward, difference, served = env.step(action)
        steps.append({'obs': obs, 'action': action, 'logp': dist.log_prob(action),
                      'reward': reward, 'difference': difference, 'served': served})
        obs = next_obs

    batch = {k: torch.stack([s[k] for s in steps]) for k in steps[0] if k != 'obs'}
    batch['obs'] = {k: torch.stack([s['obs'][k] for s in steps]) for k in steps[0]['obs']}
    return batch


"""
behaviour
    How rovers chose: the share of choices that switched POI
    since the previous step, and the share that were idle (both
    over living rovers).

Args:
    batch: dict from rollout
    idle: int index of the idle action

Returns:
    dict of float switch_rate, idle_rate
"""
def behaviour(batch, idle):
    action, alive = batch['action'], batch['obs']['alive']
    both = alive[1:] & alive[:-1]
    switched = (action[1:] != action[:-1]) & both
    return {'switch_rate': (switched.sum() / both.sum()).item(),
            'idle_rate': ((action == idle) & alive).sum().item() / alive.sum().item()}


"""
gae
    Generalized advantage estimation, per rover. The horizon is
    terminal (V = 0 after the last step).

Args:
    reward: torch.Tensor of size (T, B, A)
    value: torch.Tensor of size (T, B, A)
    gamma: float discount
    lam: float GAE lambda

Returns:
    torch.Tensor of size (T, B, A) advantages
"""
def gae(reward, value, gamma, lam):
    adv, last = torch.zeros_like(reward), 0.0
    for t in reversed(range(len(reward))):
        next_value = value[t + 1] if t + 1 < len(reward) else 0.0
        last = reward[t] + gamma * next_value - value[t] + gamma * lam * last
        adv[t] = last
    return adv


"""
update
    MAPPO update with one advantage per rover, from its own
    reward stream: its difference reward, or the shared team
    reward (--reward). Clipped policy loss over living rovers,
    entropy bonus, and per-rover value regression onto the GAE
    return. The rollout's values, and every step of a minibatch
    of episodes, are each evaluated in one batched pass.

Args:
    args: argparse.Namespace of arguments
    policy: GraphTransformer
    value_net: ValueNetwork
    optimizer: torch.optim.Optimizer
    batch: dict from rollout

Returns:
    dict of float mean losses
"""
def update(args, policy, value_net, optimizer, batch):
    with torch.no_grad():
        value = value_net(batch['obs']) * args.value_scale
    if args.reward == 'difference':
        reward = batch['difference']
    else:
        reward = batch['reward'][..., None].expand_as(value)
    adv = gae(reward, value, args.gamma, args.gae_lambda)
    target = (adv + value) / args.value_scale

    # normalize over living rover-steps
    alive = batch['obs']['alive']
    adv = (adv - adv[alive].mean()) / (adv[alive].std() + 1e-8)

    losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    n = 0
    for epoch in range(args.epochs):
        for mb in torch.randperm(adv.shape[1]).chunk(args.minibatches):
            obs = {k: v[:, mb] for k, v in batch['obs'].items()}

            # failed and empty slots contribute no policy loss
            mask = obs['alive'].float()
            dist = policy(obs)

            # clipped surrogate
            ratio = (dist.log_prob(batch['action'][:, mb]) - batch['logp'][:, mb]).exp()
            a = adv[:, mb]
            clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
            policy_loss = -(torch.min(ratio * a, clipped * a) * mask).sum() / mask.sum()
            entropy = (dist.entropy() * mask).sum() / mask.sum()
            value_loss = (value_net(obs) - target[:, mb]).pow(2).mean()
            loss = policy_loss - args.entropy_coef * entropy + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*policy.parameters(), *value_net.parameters()], args.max_grad_norm)
            optimizer.step()
            losses['policy_loss'] += policy_loss.item()
            losses['value_loss'] += value_loss.item()
            losses['entropy'] += entropy.item()
            n += 1
    return {k: v / n for k, v in losses.items()}


"""
train
    Sets up wandb logging and the checkpoint folder, builds the
    environment, both models and the Adam optimizer, resumes from a
    checkpoint if given, then runs MAPPO updates under the
    failure curriculum. Saves
    step{N}.pth every checkpoint_interval updates, and on
    interrupt.

Args:
    args: argparse.Namespace of arguments
"""
def train(args):
    # set up wandb and checkpoint path
    run_name = f"{args.name}_s{args.seed}_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = f"{args.checkpoint_dir}/{run_name}"
    wandb.init(project="marl-project", name=run_name, config=vars(args))

    torch.manual_seed(args.seed)
    env = RoverEnv(args.env_config, args.num_envs, args.comm_radius, seed=args.seed, device=device)
    policy, value_net = build_models(args, env)
    policy, value_net = policy.to(device), value_net.to(device)
    dry_run(policy, value_net, env)
    optimizer = torch.optim.Adam([
        {'params': policy.parameters(), 'lr': args.lr},
        {'params': value_net.parameters(), 'lr': args.value_lr}])
    print(f"{run_name}: {sum(p.numel() for p in policy.parameters()):,} policy parameters on {device}")

    # resume a run (as a new wandb run that continues from the checkpoint's update)
    start = 0
    if args.checkpoint:
        chkpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(chkpt['policy_state_dict'])
        value_net.load_state_dict(chkpt['value_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        start = chkpt['update']
        print(f"[resume] from update {start}")

    # save a checkpoint (the directory is made at the first save)
    def save(update_idx, name):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            'update': update_idx,
            'policy_state_dict': policy.state_dict(),
            'value_state_dict': value_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)},
            f"{ckpt_dir}/{name}.pth"
        )

    # save and exit on Ctrl-C or Slurm preemption
    state = {'update': start}
    def interrupt_handler(sig, frame):
        save(state['update'], f"step{state['update']}_int")
        wandb.finish()
        sys.exit(0)
    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)

    pbar = tqdm(range(start + 1, args.updates + 1), initial=start, total=args.updates, desc="Training", unit="update")
    for i in pbar:
        t0 = time.time()
        faults = curriculum(args, i)
        batch = rollout(env, policy, faults)
        losses = update(args, policy, value_net, optimizer, batch)
        state['update'] = i

        # training episode stats: true team return, fraction of POI-steps served, how rovers chose, curriculum
        log = {
            'env_steps': i * args.num_envs * env.horizon,
            **faults,
            'failures': env.num_faults.float().mean().item(),
            'return': batch['reward'].sum(0).mean().item(),
            'coverage': batch['served'].float().mean().item(),
            'difference': batch['difference'][batch['obs']['alive']].mean().item(),
            **behaviour(batch, env.P),
            **losses,
            'seconds': time.time() - t0}
        wandb.log(log, step=i)
        pbar.set_postfix(ret=f"{log['return']:.1f}", cov=f"{log['coverage']:.3f}", fail=f"{log['failures']:.1f}",
                         switch=f"{log['switch_rate']:.2f}", ent=f"{log['entropy']:.2f}")

        # save checkpoint every checkpoint_interval updates, and at the end
        if i % args.checkpoint_interval == 0 or i == args.updates:
            save(i, f"step{i}")

    wandb.finish()


if __name__ == "__main__":
    train(parse_args())
