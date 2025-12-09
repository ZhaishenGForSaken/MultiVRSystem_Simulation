# models/config.py
from dataclasses import dataclass

@dataclass
class DeviceParams:
    target_latency_ms: float = 16.7
    quality_min: float = 0.35
    q_rate_limit: float = 1.8
    latency_base_ms: float = 5.0
    k_latency_util: float = 60.0
    k_latency_tasks: float = 0.6
    power_base: float = 5.0
    alpha_util_to_power: float = 35.0
    pow_nl_gain: float = 0.9
    lat_nl_gain: float = 1.3
    pid_kp: float = 0.09
    pid_ki: float = 0.28
    pid_kd: float = 0.0
    jitter_sensitivity: float = 0.8
    motion_sensitivity: float = 0.5
    t_ambient: float = 40.0
    t_rise_gain: float = 0.065
    t_cool_gain: float = 0.045
    t_throttle: float = 78.0
    t_hardcap: float = 86.0
    throttle_util_cap: float = 0.65
    hardcap_util_cap: float = 0.45
    thermal_latency_penalty_ms: float = 6.0


@dataclass
class GlobalParams:
    power_budget: float = 95.0
    scheduler_period: float = 0.2
    control_dt: float = 0.01
    sim_time: float = 30.0
    num_devices: int = 2
    seed: int = 42


@dataclass
class WorkloadProfile:
    motion_amp: float = 1.0
    motion_freq: float = 0.6
    task_load_base: float = 1.0
    task_load_var: float = 0.35
    net_sync_ms: float = 0.25
    net_jitter_ms: float = 0.15
