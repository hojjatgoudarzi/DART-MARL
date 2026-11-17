"""Versioned hyperparameters; legacy package remains untouched."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class HRConfig:
    version: str = 'hr-2.2.0-event-driven'
    n_agents: int = 14
    n_pois: int = 6
    decision_interval: int = 4
    decision_mode: str = 'synchronous'  # 'asynchronous' enables per-rover observable triggers
    async_min_interval: int = 2
    chunk_length: int = 0  # 0 = exact full-sequence BPTT
    burn_in: int = 0  # rebuilds recurrent state under current weights, without gradients
    amp: bool = False  # CUDA mixed precision; off by default for numeric stability
    cuda_tf32: bool = True
    hidden: int = 128
    gamma: float = .99
    lam: float = .95
    clip: float = .2
    lr: float = 3e-4
    epochs: int = 4
    entropy: float = .02
    value_coef: float = .5
    grad_clip: float = .5
    train_episodes_per_update: int = 4
    updates: int = 256
    actor_recurrent: bool = True
    critic_recurrent: bool = True
    reset_on_reentry: bool = True
    device: str = 'cpu'
    seed: int = 101
    scenario: str = 'random_faults'
    switch_penalty: float = 0.0
    service_attribution: str = 'intent'
    # v2.3: normalized critic targets; predicted values scaled back for SMDP GAE.
    value_output_scale: float = 1.0
    # Optional ablations; zero is the raw-team-reward reference objective.
    marginal_credit_weight: float = 0.0
    potential_shaping_weight: float = 0.0
    # v2.5 TRAINING-only penalty per physically surplus participating rover-step.
    # Never applied to raw physical reward or same-information evaluations.
    overstaff_penalty_weight: float = 0.0
    # 'pivotal' reproduces the old leave-one-out term; 'shapley_service'
    # splits actual threshold-game service among all physical contributors.
    credit_mode: str = 'pivotal'
    # DEV-only checkpoint selection; final-stage raw score remains primary.
    best_dev_overstaff_weight: float = 0.0
    best_dev_checkpoint: bool = False
    idle_decision_penalty: float = 0.0
    adaptive_entropy: bool = False
    entropy_target_fraction: float = 0.55
    entropy_adapt_rate: float = 0.05
    entropy_min: float = 0.002
    entropy_max: float = 0.15
    communication: bool = False
    message_delay: int = 3
    message_dropout: float = 0.1
    message_ttl: int = 10
    message_range: float = 12.0
    curriculum: bool = False
    curriculum_mode: str = 'gated'  # 'fixed' for paired budget-equal training
    curriculum_eval_every: int = 8
    curriculum_min_updates: int = 8
    curriculum_dev_seeds: tuple[int, ...] = (700101, 700202, 700303)
    curriculum_margin_vs_random: float = 0.0
    curriculum_require_local: bool = False
    curriculum_local_fraction: float = 0.80  # at least this fraction of best local DEV heuristic
    curriculum_min_coupled_rate: float = 0.0
    curriculum_min_recovery_events: int = 0
    curriculum_min_recovery_rate: float = 0.0
    curriculum_max_stall_updates: int = 0
    deterministic_torch: bool = False
    # Research diagnostic extension: train-stage-4/5 exposure to policy-conditioned
    # recoverable service cuts. NEVER used for matched fixed-fault evaluation.
    recovery_training_fraction: float = 0.0
    recovery_intervention_step: int = 50

    def __post_init__(self):
        if self.n_agents < 1 or self.n_pois < 1 or self.decision_interval < 1 or self.hidden < 1:
            raise ValueError('invalid dimensions or interval')
        if not (0 < self.gamma <= 1 and 0 <= self.lam <= 1 and 0 < self.clip < 1):
            raise ValueError('invalid discount/PPO controls')
        if self.decision_mode not in ('synchronous', 'asynchronous') or self.async_min_interval < 1 or (self.decision_mode == 'asynchronous' and self.async_min_interval > self.decision_interval):
            raise ValueError('invalid per-agent decision policy')
        if self.curriculum_mode not in ('gated','fixed'):
            raise ValueError('unknown curriculum schedule')
        if self.chunk_length < 0 or self.burn_in < 0:
            raise ValueError('invalid chunk/burn-in length')
        if self.service_attribution not in ('intent','neutral','declared'):
            raise ValueError('invalid service attribution')
        if (self.credit_mode not in ('pivotal','shapley_service') or
                self.overstaff_penalty_weight < 0 or self.best_dev_overstaff_weight < 0):
            raise ValueError('invalid v2.5 reward/selection controls')
        if not (self.value_output_scale>0 and 0<=self.marginal_credit_weight<=1
                and self.potential_shaping_weight>=0 and self.idle_decision_penalty>=0 and 0<=self.entropy_target_fraction<=1
                and 0<=self.entropy_adapt_rate<=1 and 0<self.entropy_min<=self.entropy_max):
            raise ValueError('invalid v2.3 reward/critic/entropy controls')
        if self.switch_penalty < 0 or self.message_delay < 0 or not 0 <= self.message_dropout <= 1:
            raise ValueError('invalid communication/reward controls')
        if self.message_ttl < 1 or self.message_range <= 0 or self.curriculum_eval_every < 1 or self.curriculum_min_updates < 1 or self.curriculum_max_stall_updates < 0:
            raise ValueError('invalid curriculum/communications settings')
        if not (0 <= self.curriculum_local_fraction <= 1.5 and 0 <= self.curriculum_min_coupled_rate <= 1
                and self.curriculum_min_recovery_events >= 0 and 0 <= self.curriculum_min_recovery_rate <= 1):
            raise ValueError('invalid staged evidence gate')
        if not (0 <= self.recovery_training_fraction <= 1 and 1 <= self.recovery_intervention_step < 120):
            raise ValueError('invalid recovery-training intervention configuration')
        if self.recovery_training_fraction and self.service_attribution != 'declared':
            raise ValueError('recovery-training cuts require declared-intent service')
        if self.epochs < 1 or self.train_episodes_per_update < 1 or self.updates < 1 or self.lr <= 0:
            raise ValueError('invalid optimization settings')

    def dict(self):
        return asdict(self)
