# Methods and environment contract

## Task and action

Each of the 12 initially active rovers selects a POI index `0..P-1`, or `IDLE (-1)`. The module uses **14 tensor slots**, leaving two for replacements. The policy is **shared** but each rover has a separate hidden state. The task-conditioned categorical mask only removes tasks that are invalid under legitimately known information. The low-level controller then moves the rover toward its self-selected POI, with bounded velocity and map constraints.

## Observability and training/execution split

**Public:** fixed task coordinates, radius, nominal value, coupling requirement, skill requirements and window; each rover's own position/speed/skill and current time. **Local:** teammates within sensor radius, current local status and delayed range-limited messages when enabled. **Not Actor inputs:** entire team activity mask, remote exact positions, globally true one-shot completion when out of range, upcoming fault tape or centralized role recommendations.

The `TeamCritic` receives padded global rover/POI features during training. The Actor has **no API input** for critic features. Evaluation creates a critic instance only because the same trajectory code supports both modes; the critic is not used to choose actions.

> Attention to the distinction between source-code inspection and execution permission: the environment naturally holds global simulator state. The Actor's tensor builder is the execution firewall; tests perturb hidden remote state and check unchanged local policy outputs.

## Physical reward

A repeatable POI with threshold coupling `c` is serviced only if at least `c` **active, compatible** rovers are physically inside its radius **and have declared that task**. One-shot tasks award once within their time window. `IDLE` cannot be credited. Surplus staffing is measured in physical rover-steps over a genuinely serviced task; it may be penalized for training, but the **raw return** is always from physical service and is the evaluation metric.

## Recurrent PPO and decisions

Current training uses true sampled masked categorical actions and stores full per-agent observations, decisions, old log probabilities, team critic predictions, episode/option masks, time durations and local event triggers. PPO optimizes actor and critic with recurrent unrolling (full episode or configured chunks plus burn-in). SMDP GAE uses duration-adjusted discounting rather than assuming a fixed one-step transition. A physical failure interrupts an agent's current option; terminal and truncation must be treated differently. The saved checkpoint includes optimizer, RNG states and curriculum state for same-configuration resume.

## Curriculum and comparison groups

The 12-rover curriculum starts with simple service and increases coupling, heterogeneity and faults. The provided `pilot_cuda.yaml` uses a **fixed** five-stage schedule to preserve equal training budgets; progression is *not* proof that each stage's recovery gate was passed. Development evaluation stores gate metrics separately.

Primary comparisons use **the same public/local information** and deterministic navigator: untrained policy, Always-IDLE, random legal selection, Nearest, local coalition-aware/persistent selectors, static task-value selector, and window-aware selector. Their task selection is classical but does not invoke a runtime central allocator. Policy-conditioned disruption tests are **diagnostics**, not a valid raw-return ranking between differing chosen victims.

## High-priority unresolved question

When a feasible coupled POI loses an essential rover, why does the policy keep the same choices even after locally observable signals and legitimate redecision opportunities? The next phase is a small controlled repair-learning problem with matched fault/no-fault rollouts and an explicit gate for **correct new task assignments to the damaged POI**. Only then should we scale new GPU training on the full 12-rover environment.
