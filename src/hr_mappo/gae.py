"""Variable-duration SMDP lambda returns on synchronized macro boundaries."""
import numpy as np

def macro_gae(rewards, values, next_values, durations, terminals, gamma, lam,
              lambda_per_step=False):
    rewards=np.asarray(rewards,dtype=np.float64)
    values=np.asarray(values,dtype=np.float64)
    next_values=np.asarray(next_values,dtype=np.float64)
    durations=np.asarray(durations,dtype=np.int32)
    terminals=np.asarray(terminals,dtype=bool)
    adv=np.zeros_like(rewards)
    future=0.
    for t in range(len(rewards)-1,-1,-1):
        discount=gamma**int(durations[t])
        bootstrap=0. if terminals[t] else 1.
        delta=rewards[t]+discount*next_values[t]*bootstrap-values[t]
        # lambda per option decision, discount per elapsed physical steps.
        # Async event boundaries vary with local sensors and do not define
        # a consistent clock for GAE. Use lambda per PHYSICAL tick there.
        trace_decay = lam**int(durations[t]) if lambda_per_step else lam
        future=delta+discount*trace_decay*bootstrap*future
        adv[t]=future
    return adv.astype(np.float32),(adv+values).astype(np.float32)
