#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid CPS VR Pipeline (Fair Cooperative Scheduler)
---------------------------------------------------
- 修复 cooperative 模式下单设备占用过多资源问题
- 加入公平性惩罚与延迟平滑
- 调整 PID 控制参数以提高稳定性
- 适合 CS6376 Hybrid & Embedded Systems 课程实验展示

输出：
./outputs/
   independent/
   cooperative_2/
   cooperative_3/
   cooperative_4/
   ...
"""

import os, math, argparse
import numpy as np
import simpy
import matplotlib.pyplot as plt
from dataclasses import dataclass

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# -----------------------------
# Configurations
# -----------------------------
@dataclass
class DeviceParams:
    target_latency_ms: float = 20.0
    latency_base_ms: float = 5.0
    k_latency_util: float = 18.0
    k_latency_tasks: float = 0.5
    power_base: float = 5.0
    alpha_util_to_power: float = 25.0  # smoother power curve
    quality_min: float = 0.35
    q_rate_limit: float = 1.5
    pid_kp: float = 0.05               # slower, more stable control
    pid_ki: float = 0.15
    jitter_sensitivity: float = 0.8
    motion_sensitivity: float = 0.5


@dataclass
class GlobalParams:
    power_budget: float = 100.0
    scheduler_period: float = 0.05
    control_dt: float = 0.01
    sim_time: float = 30.0
    num_devices: int = 2
    seed: int = 42


@dataclass
class WorkloadProfile:
    motion_amp: float = 1.0
    motion_freq: float = 0.6
    task_load_base: float = 1.0
    task_load_var: float = 0.2


# -----------------------------
# Utilities
# -----------------------------
def clamp(x, lo, hi): return max(lo, min(hi, x))
def lowpass(prev, new, alpha): return alpha * new + (1 - alpha) * prev
def ensure_dir(path): os.makedirs(path, exist_ok=True)


# -----------------------------
# Device Model
# -----------------------------
class VRDevice:
    def __init__(self, env, idx, gp, dp, wp, rng):
        self.env, self.idx, self.gp, self.dp, self.wp, self.rng = env, idx, gp, dp, wp, rng
        self.q, self.q_int = 0.6, 0.0
        self.util = 0.0
        self.latency_ms = dp.latency_base_ms
        self.latency_ms_avg = dp.latency_base_ms
        self.power, self.quality = dp.power_base, 0.0
        self.clock_drift_ms = 0.0
        self.motion, self.task_load = 0.0, wp.task_load_base
        self.util_allocation = 1.0 / gp.num_devices
        self.logs = {k: [] for k in
                     ["t", "q", "util", "latency", "power", "quality", "motion", "taskload", "drift"]}
        self.proc = env.process(self.control_loop())

    def update_exogenous(self, t):
        self.motion = abs(self.wp.motion_amp * math.sin(2 * math.pi * self.wp.motion_freq * t)) \
                      + 0.1 * self.rng.standard_normal()
        self.task_load = clamp(
            self.wp.task_load_base + 0.5 * self.wp.task_load_var * math.sin(2 * math.pi * 0.1 * t)
            + 0.2 * self.rng.standard_normal(), 0.5, 1.5)

    def map_q_to_util(self, q): return clamp(0.15 + 0.85 * (q ** 1.6), 0.0, 1.0)
    def map_util_to_power(self, util): return self.dp.power_base + self.dp.alpha_util_to_power * util
    def map_states_to_latency(self, util):
        return self.dp.latency_base_ms + self.dp.k_latency_util * util + self.dp.k_latency_tasks * self.task_load

    def map_to_quality(self, latency, motion, jitter, q):
        q_term = q
        lat_penalty = 1.0 / (1.0 + 0.02 * max(0.0, latency))
        jit_penalty = 1.0 / (1.0 + self.dp.jitter_sensitivity * max(0.0, jitter))
        mot_penalty = 1.0 / (1.0 + self.dp.motion_sensitivity * max(0.0, motion))
        return clamp(q_term * lat_penalty * jit_penalty * mot_penalty, 0.0, 1.0)

    def control_update(self, dt, util_cap):
        util_demand = self.map_q_to_util(self.q)
        self.util = min(util_demand, util_cap)
        latency = self.map_states_to_latency(self.util)
        self.latency_ms_avg = lowpass(self.latency_ms_avg, latency, 0.2)
        jitter = abs(latency - self.latency_ms_avg)
        self.quality = self.map_to_quality(latency, self.motion, jitter, self.q)
        self.power = self.map_util_to_power(self.util)
        err = latency - self.dp.target_latency_ms
        self.q_int += err * dt
        dq = -(self.dp.pid_kp * err + self.dp.pid_ki * self.q_int)
        if self.quality < self.dp.quality_min and err < 0:
            dq += 0.3 * (self.dp.quality_min - self.quality)
        dq = clamp(dq, -self.dp.q_rate_limit * dt, self.dp.q_rate_limit * dt)
        self.q = clamp(self.q + dq, 0.1, 1.0)
        self.latency_ms = latency
        self.clock_drift_ms += 0.001 * (latency - self.dp.latency_base_ms) * dt

    def control_loop(self):
        dt = self.gp.control_dt
        while True:
            t = self.env.now
            self.update_exogenous(t)
            self.control_update(dt, self.util_allocation)
            for k, v in [
                ("t", t), ("q", self.q), ("util", self.util), ("latency", self.latency_ms),
                ("power", self.power), ("quality", self.quality),
                ("motion", self.motion), ("taskload", self.task_load), ("drift", self.clock_drift_ms)]:
                self.logs[k].append(v)
            yield self.env.timeout(dt)


# -----------------------------
# Fair Cooperative Scheduler
# -----------------------------
class CooperativeScheduler:
    def __init__(self, env, devices, gp, dp):
        self.env, self.devices, self.gp, self.dp = env, devices, gp, dp
        env.process(self.run())

    def run(self):
        period = self.gp.scheduler_period
        while True:
            errors = np.array(
                [lowpass(d.latency_ms_avg, d.latency_ms, 0.3) - d.dp.target_latency_ms for d in self.devices])
            priorities = np.maximum(0.2 * (errors + 5.0), 0.01)
            q_pressures = np.array([max(0.0, d.dp.quality_min - d.quality) for d in self.devices])
            prev_utils = np.array([d.util_allocation for d in self.devices])
            avg_power = np.array([
                np.mean(d.logs["power"][-500:]) if len(d.logs["power"]) > 500 else d.power
                for d in self.devices
            ])
            drift_penalty = np.array([abs(d.clock_drift_ms) for d in self.devices])

            # Fairness-aware weighted allocation
            weights = (priorities + 3.0 * q_pressures)
            weights = weights / (1.0 + 2.0 * prev_utils + 0.05 * avg_power + 0.5 * drift_penalty)

            allocs = weights / weights.sum()
            # Soft equalizer
            alloc_mean = np.mean(allocs)
            allocs = 0.8 * allocs + 0.2 * alloc_mean

            # Apply power cap
            alpha, base = self.dp.alpha_util_to_power, self.dp.power_base * len(self.devices)
            power_if_alloc = base + alpha * allocs.sum()
            if power_if_alloc > self.gp.power_budget:
                scale = max(0.1, (self.gp.power_budget - base) / (alpha * allocs.sum() + 1e-9))
                allocs *= scale

            for i, d in enumerate(self.devices):
                d.util_allocation = float(clamp(allocs[i], 0.0, 1.0))
            yield self.env.timeout(period)


# -----------------------------
# Independent Scheduler
# -----------------------------
class IndependentScheduler:
    def __init__(self, env, devices, gp, dp):
        self.env, self.devices, self.gp, self.dp = env, devices, gp, dp
        env.process(self.run())

    def run(self):
        period = self.gp.scheduler_period
        while True:
            desired = np.array([d.map_q_to_util(d.q) for d in self.devices])
            allocs = np.ones(len(self.devices)) / len(self.devices) if desired.sum() == 0 else desired / max(desired.sum(), 1e-9)
            alpha, base = self.dp.alpha_util_to_power, self.dp.power_base * len(self.devices)
            power_if_alloc = base + alpha * allocs.sum()
            if power_if_alloc > self.gp.power_budget:
                scale = max(0.1, (self.gp.power_budget - base) / (alpha * allocs.sum() + 1e-9))
                allocs *= scale
            for i, d in enumerate(self.devices):
                d.util_allocation = float(clamp(allocs[i], 0.0, 1.0))
            yield self.env.timeout(period)


# -----------------------------
# Simulation and Plotting
# -----------------------------
def run_sim(mode, gp, dp, wp, outdir):
    rng = np.random.default_rng(gp.seed)
    env = simpy.Environment()
    devices = [VRDevice(env, i, gp, dp, wp, rng) for i in range(gp.num_devices)]
    if mode == "cooperative":
        CooperativeScheduler(env, devices, gp, dp)
    elif mode == "independent":
        IndependentScheduler(env, devices, gp, dp)
    else:
        raise ValueError("Invalid mode")
    env.run(until=gp.sim_time)
    ensure_dir(outdir)
    T = np.array(devices[0].logs["t"])
    logs = {}
    for i, d in enumerate(devices):
        for k in ["latency", "power", "quality", "util", "q", "drift"]:
            logs[f"dev{i}_{k}"] = np.array(d.logs[k])
    total_power = sum(logs[f"dev{i}_power"] for i in range(gp.num_devices))

    if HAS_PANDAS:
        df = pd.DataFrame({"t": T})
        for k, v in logs.items():
            df[k] = v
        df["total_power"] = total_power
        df.to_csv(os.path.join(outdir, f"metrics_mode={mode}.csv"), index=False)

    def save_plot(fn, ylabel, cols):
        plt.figure(figsize=(9, 4.5))
        for name, arr in cols:
            plt.plot(T, arr, label=name)
        plt.xlabel("time (s)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fn), dpi=150)
        plt.close()

    save_plot("plot_latency.png", "latency (ms)",
              [(f"dev{i}", logs[f"dev{i}_latency"]) for i in range(gp.num_devices)])
    save_plot("plot_power.png", "power (W?)",
              [(f"dev{i}", logs[f"dev{i}_power"]) for i in range(gp.num_devices)] + [("total", total_power)])
    save_plot("plot_quality.png", "quality (0..1)",
              [(f"dev{i}", logs[f"dev{i}_quality"]) for i in range(gp.num_devices)])
    save_plot("plot_sync.png", "clock drift (ms)",
              [(f"dev{i}", logs[f"dev{i}_drift"]) for i in range(gp.num_devices)])
    print(f"[INFO] Finished {mode} with {gp.num_devices} devices -> {outdir}")
    return os.path.join(outdir, f"metrics_mode={mode}.csv") if HAS_PANDAS else None


# -----------------------------
# Main Batch Runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid CPS VR pipeline (fair scheduler version)")
    parser.add_argument("--sim-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    dp = DeviceParams()
    wp = WorkloadProfile()
    outroot = "./outputs"
    ensure_dir(outroot)

    # Independent baseline
    gp = GlobalParams(sim_time=args.sim_time, num_devices=2, seed=args.seed)
    run_sim("independent", gp, dp, wp, os.path.join(outroot, "independent"))

    # Cooperative multi-device
    for n in args.devices:
        gp = GlobalParams(sim_time=args.sim_time, num_devices=n, seed=args.seed)
        run_sim("cooperative", gp, dp, wp, os.path.join(outroot, f"cooperative_{n}"))

    print("\n[INFO] All simulations completed. Check ./outputs for results.")


if __name__ == "__main__":
    main()
