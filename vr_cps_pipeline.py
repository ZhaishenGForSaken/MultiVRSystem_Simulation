#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic Hybrid CPS VR Pipeline (Dynamic Power/Latency)
"""

import os, math, argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import simpy
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# ===================== 配置 =====================
@dataclass
class DeviceParams:
    target_latency_ms: float = 16.7
    quality_min: float = 0.35
    q_rate_limit: float = 1.8

    latency_base_ms: float = 5.0
    k_latency_util: float = 22.0
    k_latency_tasks: float = 0.6

    power_base: float = 5.0
    alpha_util_to_power: float = 35.0
    pow_nl_gain: float = 0.9

    lat_nl_gain: float = 1.3

    # PID
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


def clamp(x, lo, hi): return max(lo, min(hi, x))
def lowpass(prev, new, alpha): return alpha * new + (1 - alpha) * prev
def ensure_dir(p): os.makedirs(p, exist_ok=True)


class VRDevice:
    def __init__(self, env: simpy.Environment, idx: int, gp: GlobalParams, dp: DeviceParams,
                 wp: WorkloadProfile, rng: np.random.Generator, cooperative: bool, hetero: dict):
        self.env, self.idx, self.gp, self.dp, self.wp, self.rng = env, idx, gp, dp, wp, rng
        self.cooperative = cooperative

        self.ht = hetero

        self.q = clamp(0.55 + 0.1 * rng.standard_normal(), 0.3, 0.85)
        self.q_int = 0.0
        self.util = 0.0
        self.latency_ms = dp.latency_base_ms
        self.latency_ms_avg = dp.latency_base_ms
        self.power = dp.power_base
        self.quality = 0.0
        self.clock_drift_ms = 0.0
        self.temperature = dp.t_ambient + rng.uniform(-2, 2)

        self.motion = 0.0
        self.task_load = wp.task_load_base

        self.util_allocation = 1.0 / gp.num_devices

        self.logs = {k: [] for k in
                     ["t","q","util","latency","power","quality","motion","taskload","drift","temp"]}

        env.process(self.control_loop())

    def update_exogenous(self, t):
        self.motion = abs(self.wp.motion_amp * math.sin(2 * math.pi * self.wp.motion_freq * t)) \
                      + 0.1 * self.rng.standard_normal()
        base, var = self.wp.task_load_base, self.wp.task_load_var
        self.task_load = clamp(base + 0.5*var*math.sin(2*math.pi*0.12*t) + 0.25*self.rng.standard_normal(),
                               0.5, 1.8)

    def map_q_to_util(self, q) -> float:
        return clamp(self.ht["util_bias"] + (1 - self.ht["util_bias"]) * (q ** 1.6), 0.0, 1.0)

    def map_util_to_power(self, util) -> float:
        nonlinear = 1.0 + self.dp.pow_nl_gain * (util ** 2)
        load_factor = 1.0 + 0.25 * (self.task_load - 1.0)
        temp_factor = 1.0 + 0.015 * (self.temperature - self.dp.t_ambient)
        return (self.dp.power_base * self.ht["p_base_mul"]
                + self.dp.alpha_util_to_power * util * nonlinear * load_factor * temp_factor * self.ht["p_dyn_mul"])

    def map_states_to_latency(self, util) -> float:
        nonlinear = 1.0 + self.dp.lat_nl_gain * (util ** 2)
        power_effect = 1.0 + 0.003 * max(0.0, self.power - self.dp.power_base)
        L = (self.dp.latency_base_ms * self.ht["l_base_mul"] +
             self.dp.k_latency_util * util * nonlinear * power_effect * self.ht["l_util_mul"] +
             self.dp.k_latency_tasks * self.task_load * self.ht["l_task_mul"])
        return L

    def map_to_quality(self, latency_ms: float, motion: float, jitter_ms: float, q: float) -> float:
        lat_penalty = 1.0 / (1.0 + 0.02 * max(0.0, latency_ms))
        jit_penalty = 1.0 / (1.0 + self.dp.jitter_sensitivity * max(0.0, jitter_ms))
        mot_penalty = 1.0 / (1.0 + self.dp.motion_sensitivity * max(0.0, motion))
        return clamp(q * lat_penalty * jit_penalty * mot_penalty, 0.0, 1.0)

    def thermal_update(self, dt: float):
        dT = self.dp.t_rise_gain * self.power - self.dp.t_cool_gain * (self.temperature - self.dp.t_ambient)
        self.temperature += dT * (dt * 60.0)
        self.temperature = clamp(self.temperature, self.dp.t_ambient - 5.0, 100.0)

    def thermal_caps(self) -> Tuple[float, float]:
        if self.temperature >= self.dp.t_hardcap:
            return self.dp.hardcap_util_cap, self.dp.thermal_latency_penalty_ms * 2.0
        if self.temperature >= self.dp.t_throttle:
            return self.dp.throttle_util_cap, self.dp.thermal_latency_penalty_ms
        return 1.0, 0.0

    def control_update(self, dt: float):
        util_cap_sched = self.util_allocation
        util_cap_therm, extra_lat_ms = self.thermal_caps()

        util_demand = self.map_q_to_util(self.q)
        self.util = min(util_demand, util_cap_sched, util_cap_therm)

        self.power = self.map_util_to_power(self.util)
        latency = self.map_states_to_latency(self.util)

        if self.cooperative:
            latency += self.wp.net_sync_ms + self.rng.standard_normal() * self.wp.net_jitter_ms

        latency += extra_lat_ms

        self.latency_ms_avg = lowpass(self.latency_ms_avg, latency, alpha=0.3)
        jitter = abs(latency - self.latency_ms_avg)

        self.quality = self.map_to_quality(latency, self.motion, jitter, self.q)

        err = latency - self.dp.target_latency_ms
        self.q_int += err * dt
        dq = -(self.dp.pid_kp * err + self.dp.pid_ki * self.q_int)
        if self.quality < self.dp.quality_min and err < 0:
            dq += 0.35 * (self.dp.quality_min - self.quality)

        dq = clamp(dq, -self.dp.q_rate_limit * dt, self.dp.q_rate_limit * dt)
        self.q = clamp(self.q + dq, 0.1, 1.0)
        self.latency_ms = latency

        self.clock_drift_ms += 0.001 * (latency - self.dp.latency_base_ms) * dt
        self.thermal_update(dt)

    def control_loop(self):
        dt = self.gp.control_dt
        while True:
            t = self.env.now
            self.update_exogenous(t)
            self.control_update(dt)

            for k, v in [("t", t), ("q", self.q), ("util", self.util), ("latency", self.latency_ms),
                         ("power", self.power), ("quality", self.quality), ("motion", self.motion),
                         ("taskload", self.task_load), ("drift", self.clock_drift_ms), ("temp", self.temperature)]:
                self.logs[k].append(v)

            yield self.env.timeout(dt)


class CooperativeScheduler:
    def __init__(self, env, devices: List[VRDevice], gp: GlobalParams, dp: DeviceParams):
        self.env, self.devices, self.gp, self.dp = env, devices, gp, dp
        env.process(self.run())

    def run(self):
        while True:
            lat_err = np.array([d.latency_ms - d.dp.target_latency_ms for d in self.devices])
            need = np.maximum(lat_err, 0.0)
            q_pressure = np.array([max(0.0, d.dp.quality_min - d.quality) for d in self.devices])
            therm_penalty = np.array([1.0 if d.temperature < d.dp.t_throttle else 0.6 for d in self.devices])

            weights = (1.0 + need) + 2.0 * q_pressure
            weights *= therm_penalty

            if weights.sum() <= 1e-9:
                allocs = np.ones(len(self.devices)) / len(self.devices)
            else:
                allocs = weights / weights.sum()

            base = sum(d.dp.power_base * d.ht["p_base_mul"] for d in self.devices)
            alpha = self.dp.alpha_util_to_power
            est_dyn = sum(alpha * a * (1.0 + self.dp.pow_nl_gain * (a ** 2)) * d.ht["p_dyn_mul"] for a, d in zip(allocs, self.devices))
            power_if_alloc = base + est_dyn

            if power_if_alloc > 1.15 * self.gp.power_budget:
                scale = max(0.6, (self.gp.power_budget - base) / (est_dyn + 1e-9))
                allocs *= scale

            for i, d in enumerate(self.devices):
                d.util_allocation = float(clamp(allocs[i], 0.0, 1.0))

            yield self.env.timeout(self.gp.scheduler_period)


class IndependentScheduler:
    def __init__(self, env, devices: List[VRDevice], gp: GlobalParams, dp: DeviceParams):
        self.env, self.devices, self.gp, self.dp = env, devices, gp, dp
        env.process(self.run())

    def run(self):
        while True:
            desired = np.array([d.map_q_to_util(d.q) for d in self.devices])
            if desired.sum() <= 1e-9:
                allocs = np.zeros_like(desired)
            else:
                allocs = desired / desired.sum()

            base = sum(d.dp.power_base * d.ht["p_base_mul"] for d in self.devices)
            alpha = self.dp.alpha_util_to_power
            est_dyn = sum(alpha * a * (1.0 + self.dp.pow_nl_gain * (a ** 2)) * d.ht["p_dyn_mul"] for a, d in zip(allocs, self.devices))
            power_if_alloc = base + est_dyn

            if power_if_alloc > 1.15 * self.gp.power_budget:  # ☆软上限
                scale = max(0.6, (self.gp.power_budget - base) / (est_dyn + 1e-9))
                allocs *= scale

            for a, d in zip(allocs, self.devices):
                d.util_allocation = float(clamp(a, 0.0, 1.0))

            yield self.env.timeout(self.gp.scheduler_period)


def draw_heterogeneity(rng: np.random.Generator):
    return {
        "p_base_mul": float(np.clip(rng.normal(1.0, 0.07), 0.85, 1.2)),
        "p_dyn_mul": float(np.clip(rng.normal(1.0, 0.10), 0.8, 1.25)),
        "l_base_mul": float(np.clip(rng.normal(1.0, 0.06), 0.85, 1.2)),
        "l_util_mul": float(np.clip(rng.normal(1.0, 0.08), 0.8, 1.25)),
        "l_task_mul": float(np.clip(rng.normal(1.0, 0.08), 0.8, 1.25)),
        "util_bias": float(np.clip(rng.normal(0.12, 0.03), 0.05, 0.2)),
    }

def run_sim(mode: str, gp: GlobalParams, dp: DeviceParams, wp: WorkloadProfile, outdir: str):
    rng = np.random.default_rng(gp.seed)
    env = simpy.Environment()

    cooperative = (mode == "cooperative")
    devices: List[VRDevice] = []
    for i in range(gp.num_devices):
        hetero = draw_heterogeneity(rng)
        devices.append(VRDevice(env, i, gp, dp, wp, rng, cooperative=cooperative, hetero=hetero))

    if cooperative:
        CooperativeScheduler(env, devices, gp, dp)
    else:
        IndependentScheduler(env, devices, gp, dp)

    env.run(until=gp.sim_time)
    ensure_dir(outdir)

    T = np.array(devices[0].logs["t"])
    logs = {}
    for i, d in enumerate(devices):
        for k in ["latency", "power", "quality", "util", "q", "drift", "temp"]:
            logs[f"dev{i}_{k}"] = np.array(d.logs[k])
    total_power = sum(logs[f"dev{i}_power"] for i in range(gp.num_devices))

    if HAS_PANDAS:
        df = pd.DataFrame({"t": T})
        for k, v in logs.items(): df[k] = v
        df["total_power"] = total_power
        df.to_csv(os.path.join(outdir, f"metrics_mode={mode}.csv"), index=False)

    def save_plot(fn, ylabel, series: List[Tuple[str, np.ndarray]]):
        plt.figure(figsize=(9, 4.5))
        for name, y in series:
            plt.plot(T, y, label=name)
        plt.xlabel("time (s)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fn), dpi=150)
        plt.close()

    save_plot("plot_latency.png", "latency (ms)",
              [(f"dev{i}", logs[f"dev{i}_latency"]) for i in range(gp.num_devices)])
    save_plot("plot_power.png", "power (W)",
              [(f"dev{i}", logs[f"dev{i}_power"]) for i in range(gp.num_devices)] + [("total", total_power)])
    save_plot("plot_quality.png", "quality (0..1)",
              [(f"dev{i}", logs[f"dev{i}_quality"]) for i in range(gp.num_devices)])
    save_plot("plot_sync.png", "clock drift (ms)",
              [(f"dev{i}", logs[f"dev{i}_drift"]) for i in range(gp.num_devices)])
    save_plot("plot_temperature.png", "temperature (°C)",
              [(f"dev{i}", logs[f"dev{i}_temp"]) for i in range(gp.num_devices)])

    print(f"[INFO] Finished {mode} with {gp.num_devices} devices -> {outdir}")


# ===================== 批量入口 =====================
def main():
    parser = argparse.ArgumentParser(description="Realistic Dynamic CPS VR Simulator")
    parser.add_argument("--sim-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--power-budget", type=float, default=95.0)
    parser.add_argument("--sched-period", type=float, default=0.2, help="coordinator period (s)")
    args = parser.parse_args()

    dp = DeviceParams()
    wp = WorkloadProfile()
    outroot = "./outputs"
    ensure_dir(outroot)

    gp = GlobalParams(sim_time=args.sim_time, num_devices=2, seed=args.seed,
                      power_budget=args.power_budget, scheduler_period=args.sched_period)
    run_sim("independent", gp, dp, wp, os.path.join(outroot, "independent"))

    for n in args.devices:
        gp = GlobalParams(sim_time=args.sim_time, num_devices=n, seed=args.seed,
                          power_budget=args.power_budget, scheduler_period=args.sched_period)
        run_sim("cooperative", gp, dp, wp, os.path.join(outroot, f"cooperative_{n}"))

    print("\n[INFO] All simulations completed. Check ./outputs/ ...")

if __name__ == "__main__":
    main()
