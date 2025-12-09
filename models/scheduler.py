# models/scheduler.py
import numpy as np
from models.utils import clamp

# ============================================================
# Cooperative Scheduler
# ============================================================
class CooperativeScheduler:
    """
    Cooperative mode: scheduler redistributes utilization
    based on latency error, quality pressure, and thermal penalty.
    """
    def __init__(self, env, devices, gp, dp):
        self.env, self.devices, self.gp, self.dp = env, devices, gp, dp
        env.process(self.run())

    def run(self):
        while True:
            # Skip offline devices
            active_devices = [d for d in self.devices if not getattr(d, "offline", False)]
            if len(active_devices) == 0:
                yield self.env.timeout(self.gp.scheduler_period)
                continue

            # rewrite arrays for active devices only
            devices = active_devices

            # latency error
            lat_err = np.array([d.latency_ms - d.dp.target_latency_ms for d in devices])
            # quality pressure
            q_pressure = np.array([max(0.0, d.dp.quality_min - d.quality) for d in devices])
            # temperature penalty (reduce allocation for hot devices)
            therm_penalty = np.array([1.0 if getattr(d, "temperature", 40) < d.dp.t_throttle else 0.6 for d in devices])

            weights = (1.0 + np.maximum(lat_err, 0.0)) + 2.0 * q_pressure
            weights *= therm_penalty

            if weights.sum() <= 1e-9:
                allocs = np.ones(len(devices)) / len(devices)
            else:
                allocs = weights / weights.sum()

            # estimate total power and apply soft power budget
            base = sum(d.dp.power_base for d in devices)
            alpha = self.dp.alpha_util_to_power
            est_dyn = sum(alpha * a * (1.0 + self.dp.pow_nl_gain * (a ** 2)) for a in allocs)
            power_if_alloc = base + est_dyn

            if power_if_alloc > 1.1 * self.gp.power_budget:
                scale = max(0.5, (self.gp.power_budget - base) / (est_dyn + 1e-9))
                allocs *= scale

            # apply allocations
            for i, d in enumerate(devices):
                d.util_allocation = float(clamp(allocs[i], 0.0, 1.0))

            yield self.env.timeout(self.gp.scheduler_period)


# ============================================================
# Independent Scheduler
# ============================================================
class IndependentScheduler:
    """
    Independent mode: each device greedily demands utilization
    based on its local q (quality) controller.
    If total estimated power exceeds the budget, scale all equally.
    """
    def __init__(self, env, devices, gp, dp):
        self.env, self.devices, self.gp, self.dp = env, devices, gp, dp
        env.process(self.run())

    def run(self):
        while True:
            desired = np.array([d.map_q_to_util(d.q) for d in self.devices])
            if desired.sum() <= 1e-9:
                allocs = np.zeros_like(desired)
            else:
                allocs = desired / desired.sum()

            # power estimation and global soft budget
            base = sum(d.dp.power_base for d in self.devices)
            alpha = self.dp.alpha_util_to_power
            est_dyn = sum(alpha * a * (1.0 + self.dp.pow_nl_gain * (a ** 2)) for a in allocs)
            power_if_alloc = base + est_dyn

            if power_if_alloc > 1.1 * self.gp.power_budget:
                scale = max(0.5, (self.gp.power_budget - base) / (est_dyn + 1e-9))
                allocs *= scale

            for a, d in zip(allocs, self.devices):
                d.util_allocation = float(clamp(a, 0.0, 1.0))

            yield self.env.timeout(self.gp.scheduler_period)
