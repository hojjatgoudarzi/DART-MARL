import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml


"""
RoverEnv
    Batched multi-rover POI allocation with rover failures and
    limited communication. B worlds step in lockstep, each rover
    slot is one agent, and the team shares one reward. Each rover
    picks a POI (or idle) every step, and a fixed controller
    drives it straight to that POI's centre. Rovers fail
    permanently: each world gets a number of failure steps on
    reset, and at each one a living rover fails, picked at random
    or by a greedy adversary (the rover whose loss costs the team
    the most right then).
"""
class RoverEnv:
    """
    RoverEnv.__init__
        Loads a yaml config (see configs/): the world and the POIs.

        Args:
            config_path: string path to yaml config
            num_envs: int number of parallel worlds B
            comm_radius: float communication radius in map units (inf = everyone)
            seed: int seed for spawns and failures
            device: torch.device or string
    """
    def __init__(self, config_path, num_envs, comm_radius=math.inf, seed=0, device="cpu"):
        cfg = yaml.safe_load(open(config_path))
        pois = cfg["pois"]
        self.B, self.device = num_envs, device
        self.rng = np.random.default_rng(seed)
        self.horizon = cfg["horizon"]
        self.margin = cfg["fault_margin"]
        self.speed = cfg["speed"]
        self.comm_radius = comm_radius
        self.A, self.P = cfg["rovers"], len(pois)
        self.dims = torch.tensor(cfg["dimensions"], dtype=torch.float, device=device)

        # points of interest, one copy per world (so maps can later be randomized per world)
        def per_world(key, dtype):
            x = torch.tensor([p[key] for p in pois], dtype=dtype, device=device)
            return x.expand(self.B, *x.shape).clone()
        self.poi_pos = per_world("position", torch.float)
        self.radius = per_world("radius", torch.float)
        self.coupling = per_world("coupling", torch.long)
        self.value = per_world("value", torch.float)
        self.window = per_world("window", torch.long)
        self.repeat = per_world("repeat", torch.bool)

        # rover features: pos (2), time; POI features: see observe()
        self.agent_dim = 3
        self.poi_dim = 6

    """
    RoverEnv.reset
        Starts a new episode in every world, and schedules its
        failures: a count drawn uniformly from [min_faults,
        max_faults], at distinct steps away from the episode's ends.
        Each failure is adversarial with probability `adversarial`,
        and random otherwise.

        Args:
            min_faults: int fewest failures per episode
            max_faults: int most failures per episode (at most rovers - 1)
            adversarial: float probability each failure is adversarial

        Returns:
            dict of observation tensors (see observe)
    """
    def reset(self, min_faults=0, max_faults=0, adversarial=0.0):
        B, H, A = self.B, self.horizon, self.A

        # rovers start uniformly over the map (drawn first, so the same seed gives the same starts for any failures)
        pos = self.rng.uniform(0, 1, (B, A, 2)) * self.dims.cpu().numpy()
        self.pos = torch.tensor(pos, dtype=torch.float, device=self.device)

        steps = np.arange(self.margin, H - self.margin)
        counts = self.rng.integers(min_faults, max(min_faults, max_faults) + 1, B)
        counts = counts.clip(max=min(A - 1, len(steps)))
        self.fault_step = torch.full((B, H), False)
        for b, n in enumerate(counts):
            self.fault_step[b, torch.as_tensor(self.rng.choice(steps, n, replace=False))] = True
        self.fault_step = self.fault_step.to(self.device)
        self.adversarial = torch.tensor(self.rng.uniform(0, 1, (B, H)) < adversarial, device=self.device)
        self.num_faults = torch.tensor(counts, device=self.device)
        self.active = torch.ones(B, A, dtype=torch.bool, device=self.device)
        self.completed = torch.zeros(B, self.P, dtype=torch.bool, device=self.device)
        self.t = 0
        return self.observe()

    """
    RoverEnv.fail
        Fails one living rover in every world with a failure
        scheduled at the current step. A random failure picks
        uniformly; an adversarial one picks the rover with the
        largest difference reward on the last step (random among
        ties, and fully random if no rover was critical).

        Args:
            difference: torch.Tensor of size (B, A) last step's difference rewards
    """
    def fail(self, difference):
        due = self.fault_step[:, self.t]
        if not due.any():
            return
        noise = torch.tensor(self.rng.uniform(0, 1, (self.B, self.A)), dtype=torch.float, device=self.device)
        targeted = self.adversarial[:, self.t] & (difference * self.active).amax(-1).gt(0)
        score = torch.where(targeted[:, None], difference + 1e-3 * noise, noise)
        victim = score.masked_fill(~self.active, -math.inf).argmax(-1)
        self.active[due, victim[due]] = False

    """
    RoverEnv.step
        Controller: every active rover moves straight toward its
        chosen POI's centre at full speed and stops there (idle
        rovers stay put). Then pays every served POI, and applies
        the next step's failures. The difference reward of a rover
        is how much the team reward would drop without it (0 for a
        rover that is redundant or failed).

        Args:
            actions: torch.Tensor of size (B, A) in [0, P], P = idle

        Returns:
            dict of next observation tensors
            torch.Tensor of size (B,) team reward
            torch.Tensor of size (B, A) difference rewards
            torch.Tensor of size (B, P) bool POIs served this step
    """
    def step(self, actions):
        idle = actions == self.P
        target = self.poi_pos.gather(1, actions.clamp(max=self.P - 1)[..., None].expand(-1, -1, 2))
        target = torch.where(idle[..., None], self.pos, target)
        delta = target - self.pos
        scale = (self.speed / delta.norm(dim=-1).clamp(min=1e-12)).clamp(max=1)
        self.pos = self.pos + delta * (scale * self.active)[..., None]

        # team reward, and each rover's difference reward D_i = G - G_{-i}
        served, served_without = self.coverage()
        reward = (served * self.value).sum(-1)
        difference = reward[:, None] - (served_without * self.value[:, None]).sum(-1)

        self.completed |= served & ~self.repeat
        self.t += 1
        if self.t < self.horizon:
            self.fail(difference)
        return self.observe(), reward, difference, served

    """
    RoverEnv.available
        POIs that can still pay: not a completed one-shot POI and
        not past its window. POIs whose window has not opened yet
        are available, since arriving early is useful.

        Returns:
            torch.Tensor of size (B, P) bool
    """
    def available(self):
        return ~self.completed & (self.t <= self.window[..., 1])

    """
    RoverEnv.coverage
        A rover covers the nearest (by distance / radius) open POI
        it is inside, and counts for that POI only. A POI is served
        when at least `coupling` rovers cover it. Also returns what
        would be served if each rover were absent: since a rover's
        choice of POI doesn't depend on the others, removing it only
        lowers its own POI's count.

        Returns:
            torch.Tensor of size (B, P) bool served
            torch.Tensor of size (B, A, P) bool served without rover i
    """
    def coverage(self):
        open_ = (self.window[..., 0] <= self.t) & self.available()
        dist = torch.cdist(self.pos, self.poi_pos)
        inside = self.active[..., None] & open_[:, None] & (dist < self.radius[:, None])
        choice = torch.where(inside, dist / self.radius[:, None], math.inf).argmin(-1)

        # covers[b, i, p]: rover i counts for POI p
        covers = F.one_hot(choice, self.P).float() * inside.any(-1, keepdim=True)
        counts = covers.sum(1)
        served = open_ & (counts >= self.coupling)
        served_without = open_[:, None] & (counts[:, None] - covers >= self.coupling[:, None])
        return served, served_without

    """
    RoverEnv.communication
        Which rovers hear each other: both active and closer than
        the comm radius.

        Returns:
            torch.Tensor of size (B, A, A) bool, False on the diagonal
    """
    def communication(self):
        in_range = torch.cdist(self.pos, self.pos) < self.comm_radius
        alive = self.active[:, :, None] & self.active[:, None]
        return alive & in_range & ~torch.eye(self.A, dtype=torch.bool, device=self.device)

    """
    RoverEnv.observe
        Token observations. Positions are fractions of the map, and
        policies take them relative to each rover. Rows of failed
        rovers are zero.

        Returns:
            dict of
                agent: (B, A, agent_dim) own pos, time
                poi: (B, P, poi_dim) value, coupling, repeat, steps until
                    open / H, steps until close / H, available
                pos: (B, A, 2) rover positions
                poi_pos: (B, P, 2) POI positions
                alive: (B, A) bool active rovers
                comm: (B, A, A) bool who hears whom
                action_mask: (B, P + 1) bool selectable POIs, then idle
    """
    def observe(self):
        H, available = self.horizon, self.available()
        time = torch.full((self.B, self.A, 1), self.t / H, device=self.device)
        agent = torch.cat([self.pos / self.dims, time], dim=-1) * self.active[..., None]
        poi = torch.stack([
            self.value,
            self.coupling.float(),
            self.repeat.float(),
            (self.window[..., 0] - self.t).clamp(min=0) / H,
            (self.window[..., 1] - self.t).clamp(min=0) / H,
            available.float()], dim=-1)
        return {
            'agent': agent,
            'poi': poi,
            'pos': self.pos / self.dims,
            'poi_pos': self.poi_pos / self.dims,
            'alive': self.active.clone(),
            'comm': self.communication(),
            'action_mask': torch.cat([available, torch.ones(self.B, 1, dtype=torch.bool, device=self.device)], dim=-1)}
