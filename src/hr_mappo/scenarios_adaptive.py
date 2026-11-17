"""v2.4 task-allocation challenge: pre-registered, policy-INDEPENDENT events.

Twelve mobile rovers (six of each skill) and two unused slots have four
repeatable 3-rover tasks (two per skill). Both tasks for a skill have
OVERLAPPING windows, so forming two separate coalitions is useful but a
post-fault team with five eligible rovers cannot sustain both. Early/late
windows and POI indices are independently permuted each episode. Values
are comparable, not the permanently dominant index-2/index-3 structure
in v2.3. Real transfer is required to respond to a damaged coalition.

All window metadata are public. Future faults are NOT part of observations.
The geometry, event times, and affected slots are sampled before rollout;
no policy's realized coalition determines a fault target. This creates
opportunities, not guaranteed service disruptions for *every* policy.
"""
from dataclasses import replace
import numpy as np
from hr_mappo.disruptions.events import DisruptionEvent


def scenario_v24(base, seed: int, *, heldout=False, faults=True, stress=False):
    if len(base.rovers) != 12 or base.max_agents != 14 or len(base.pois) != 6:
        raise ValueError('v2.4 requires 12 initially active, two reserve slots, six POIs')
    g = np.random.default_rng(int(seed))
    # Within-skill centers are separated enough to require travel between
    # tasks, yet near enough for recovery before the competing window closes.
    centers = np.array([[11., 11.], [11., 29.], [27., 11.], [27., 29.],
                        [20., 17.], [20., 23.]], dtype=np.float64)
    if heldout:
        centers = np.array([[10., 14.], [14., 29.], [29., 12.], [27., 29.],
                            [19., 7.], [22., 33.]], dtype=np.float64)
    # D4 transforms applied consistently to all POIs and initial cohorts.
    for _ in range(int(g.integers(4))):
        centers = np.column_stack([40.-centers[:, 1], centers[:, 0]])
    if g.integers(2):
        centers[:, 0] = 40.-centers[:, 0]
    centers = np.clip(centers+g.uniform(-1.4 if not heldout else -2.4,
                                        1.4 if not heldout else 2.4, (6, 2)), 3.5, 36.5)
    flip = int(g.integers(2))
    skill = {0: flip, 2: flip, 1: 1-flip, 3: 1-flip}
    early = {}
    for pair in ((0, 2), (1, 3)):
        early[pair] = int(g.choice(pair))
    pois = []
    for k, old in enumerate(base.pois):
        if k < 4:
            pair = (0, 2) if k in (0, 2) else (1, 3)
            is_early = k == early[pair]
            # Both tasks can earn reward in the mid-episode overlap.
            # Deadlines force selection/reallocation; fixed index choices
            # cannot win uniformly over randomized early/late identities.
            window = (0, int(g.integers(74, 86))) if is_early else (int(g.integers(27, 36)), 119)
            # Similar total service value; both groups matter pre-fault.
            value = float((4.5 if is_early else 4.7)*g.uniform(.88, 1.12))
            coupling, required, repeat, radius = 3, skill[k], True, 3.1
        else:
            window = (int(g.integers(17, 28)), int(g.integers(57, 72))) if k==4 else (int(g.integers(56, 68)), int(g.integers(96, 109)))
            value = float(g.uniform(8., 12.))
            coupling, required, repeat, radius = 1, None, False, 2.6
        pois.append(replace(old, position=tuple(centers[k]), radius=radius,
                            coupling=coupling, required_skill=required,
                            repeat=repeat, window=window, value=value))
    # Before any faults, exactly three candidate rovers start nearer to
    # each same-skill POI. Every start is OUTSIDE every service circle.
    roster = []
    for target in (0, 1, 2, 3):
        for _ in range(3):
            for attempt in range(30000):
                angle = float(g.uniform(0,2*np.pi))
                dist = float(g.uniform(6.8,9.3))
                candidate = np.clip(centers[target] + dist*np.array([np.cos(angle), np.sin(angle)]), 1.2, 38.8)
                if any(np.linalg.norm(candidate - c) < p.radius+2.2 for c,p in zip(centers,pois)):
                    continue
                if any(np.linalg.norm(candidate - r[0]) < 1.0 for r in roster):
                    continue
                roster.append((candidate,skill[target],target))
                break
            else:
                raise RuntimeError('could not construct nontrivial v2.4 initial state')
    g.shuffle(roster)
    rovers = tuple(replace(old, start=tuple(item[0]), skill=int(item[1]),
                            speed=float(g.uniform(.95,1.35)),
                            sensor_radius=float(g.uniform(11.0,17.5)))
                   for old,item in zip(base.rovers,roster))
    events = ()
    if faults:
        # Choose pre-registered initial cohorts, not currently chosen POI.
        # Each skill loses one member during BOTH-task overlap. 5 remaining
        # agents can restore either coupling-3 service, but cannot serve both.
        first_pool = [i for i,r in enumerate(rovers) if r.skill==flip]
        second_pool = [i for i,r in enumerate(rovers) if r.skill==1-flip]
        victim_a = int(g.choice(first_pool)); victim_b = int(g.choice(second_pool))
        other_a = int(g.choice([i for i in first_pool if i!=victim_a]))
        other_b = int(g.choice([i for i in second_pool if i!=victim_b]))
        def jitter():return int(g.integers(-2,3))
        # A replacement event removes a different original member while
        # activating slot 12; net active count is unchanged for its skill.
        spawn = tuple(np.clip(centers[int(roster[other_a][2])] + g.uniform(-6,6,2), 2.,38.))
        events = (
            DisruptionEvent(step=49+jitter(),kind='permanent_failure',rover=victim_a,detection_delay=5),
            DisruptionEvent(step=61+jitter(),kind='temporary_disable',rover=victim_b,
                            duration=42 if not stress else 53,detection_delay=6),
            DisruptionEvent(step=83+jitter(),kind='replacement_agent',rover=other_a,
                            new_slot=12,spawn=spawn,detection_delay=4),
            DisruptionEvent(step=99+jitter(),kind='capability_degradation',rover=other_b,
                            speed_factor=.45,sensor_factor=.65,detection_delay=5),
        )
    scene = replace(base,name=f'hr_v24_{"heldout" if heldout else "dev"}{"_stress" if stress else ""}_{seed}',
                    rovers=rovers,pois=tuple(pois),events=events,spawn_jitter=0.)
    # Feasible remediation after the initial loss, even without returning agent.
    assert all(sum(r.skill==task.required_skill for r in rovers)-1>=task.coupling
               for task in scene.pois[:4])
    return scene
