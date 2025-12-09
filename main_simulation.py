#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation entry for multi-device VR CPS system
----------------------------------------------------
- Uses modularized components from ./models/
- Generates full plots: latency, power, quality, sync, temperature
- Saves per-device metrics + total power CSV
"""

import os, argparse
import numpy as np
import simpy
import matplotlib.pyplot as plt
from models.config import DeviceParams, GlobalParams, WorkloadProfile
from models.device_model import VRDevice
from models.scheduler import CooperativeScheduler, IndependentScheduler
from models.utils import ensure_dir, draw_heterogeneity

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# ---------------------------
# Simulation Core
# ---------------------------
def run_sim(mode: str, num_devices: int, sim_time: float, outdir: str, power_budget: float = 95.0):
    """Run simulation and generate plots (following previous logic)."""
    rng = np.random.default_rng(42)
    env = simpy.Environment()

    dp, wp = DeviceParams(), WorkloadProfile()
    gp = GlobalParams(sim_time=sim_time, num_devices=num_devices, power_budget=power_budget)

    cooperative = (mode == "cooperative")
    devices = [VRDevice(env, i, gp, dp, wp, rng, cooperative, draw_heterogeneity(rng))
               for i in range(num_devices)]

    if cooperative:
        CooperativeScheduler(env, devices, gp, dp)
    else:
        IndependentScheduler(env, devices, gp, dp)

    env.run(until=gp.sim_time)
    ensure_dir(outdir)

    # ----- collect logs -----
    T = np.array(devices[0].logs["t"])
    logs = {}
    for i, d in enumerate(devices):
        for k in ["latency", "power", "quality", "util", "q", "temp"]:
            logs[f"dev{i}_{k}"] = np.array(d.logs[k])
    total_power = sum(logs[f"dev{i}_power"] for i in range(num_devices))

    # ----- save CSV -----
    if HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame({"t": T})
        for k, v in logs.items():
            df[k] = v
        df["total_power"] = total_power
        df.to_csv(os.path.join(outdir, f"metrics_mode={mode}.csv"), index=False)

    # ----- plotting (same as your old style) -----
    def save_plot(fn, ylabel, series):
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

    # latency
    save_plot("plot_latency.png", "latency (ms)",
              [(f"dev{i}", logs[f"dev{i}_latency"]) for i in range(num_devices)])

    # power
    save_plot("plot_power.png", "power (W)",
              [(f"dev{i}", logs[f"dev{i}_power"]) for i in range(num_devices)] + [("total", total_power)])

    # quality
    save_plot("plot_quality.png", "quality (0..1)",
              [(f"dev{i}", logs[f"dev{i}_quality"]) for i in range(num_devices)])

    # utilization (q)
    save_plot("plot_utilization.png", "GPU utilization",
              [(f"dev{i}", logs[f"dev{i}_util"]) for i in range(num_devices)])

    # temperature
    save_plot("plot_temperature.png", "temperature (°C)",
              [(f"dev{i}", logs[f"dev{i}_temp"]) for i in range(num_devices)])

    print(f"[INFO] Finished {mode} simulation with {num_devices} devices → {outdir}")


# ---------------------------
# Batch launcher
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-device VR CPS Simulation")
    parser.add_argument("--sim-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--power-budget", type=float, default=95.0)
    args = parser.parse_args()

    outroot = "./outputs"
    ensure_dir(outroot)

    # independent baseline (2 devices)
    run_sim("independent", num_devices=2, sim_time=args.sim_time,
            outdir=os.path.join(outroot, "independent"), power_budget=args.power_budget)

    # cooperative multi-device
    for n in args.devices:
        run_sim("cooperative", num_devices=n, sim_time=args.sim_time,
                outdir=os.path.join(outroot, f"cooperative_{n}"), power_budget=args.power_budget)

    print("\n[INFO] All simulations completed. Results saved in ./outputs/")


if __name__ == "__main__":
    main()
