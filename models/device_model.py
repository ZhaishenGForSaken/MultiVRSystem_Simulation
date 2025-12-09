# models/device_model.py
import math
import numpy as np
import simpy
from models.utils import clamp, lowpass

class VRDevice:
    def __init__(self, env, idx, gp, dp, wp, rng, cooperative, hetero):
        self.env, self.idx, self.gp, self.dp, self.wp, self.rng = env, idx, gp, dp, wp, rng
        self.cooperative = cooperative
        self.ht = hetero

        self.q = clamp(0.55 + 0.1 * rng.standard_normal(), 0.3, 0.85)
        self.q_int, self.util = 0.0, 0.0
        self.latency_ms = dp.latency_base_ms
        self.latency_ms_avg = dp.latency_base_ms
        self.power, self.quality = dp.power_base, 0.0
        self.temperature = dp.t_ambient + rng.uniform(-2, 2)
        self.motion, self.task_load = 0.0, wp.task_load_base
        self.util_allocation = 1.0 / gp.num_devices
        self.logs = {k: [] for k in
                     ["t","q","util","latency","power","quality","motion","taskload","temp"]}

        self.external_load_factor = 1.0
        self.offline = False

        env.process(self.control_loop())

    def update_exogenous(self, t):
        self.motion = abs(self.wp.motion_amp * math.sin(2 * math.pi * (self.wp.motion_freq + 0.05*self.idx) * t)) \
                      + 0.15 * self.rng.standard_normal()
        base, var = self.wp.task_load_base, self.wp.task_load_var
        bias = 0.3 * (self.idx / max(1, self.gp.num_devices - 1))
        self.task_load = clamp(base + bias + 0.6*var*math.sin(2*math.pi*0.1*t) +
                               0.25*self.rng.standard_normal(), 0.5, 2.0)
        self.task_load *= self.external_load_factor
        self.task_load = clamp(self.task_load, 0.3, 5.0)

    def map_q_to_util(self, q): return clamp(self.ht["util_bias"] + (1 - self.ht["util_bias"]) * (q**1.6), 0.0, 1.0)

    def map_util_to_power(self, util) -> float:
        """
        Nonlinear power model with strong temperature feedback.
        Power increases with utilization and load,
        but at high temperature, efficiency drops (thermal throttling).
        """
        nonlinear = 1.0 + self.dp.pow_nl_gain * (util ** 2)
        load_factor = 1.0 + 0.3 * (self.task_load - 1.0)
        temp_diff = self.temperature - self.dp.t_ambient
        if temp_diff < 0:
            temp_factor = 1.0
        elif temp_diff < 20:
            temp_factor = 1.0 + 0.03 * temp_diff
        else:
            temp_factor = 1.6 - 0.02 * min(temp_diff, 40)

        base_p = self.dp.power_base * self.ht["p_base_mul"]
        dyn_p = (self.dp.alpha_util_to_power * util * nonlinear *
                 load_factor * temp_factor * self.ht["p_dyn_mul"])

        power = base_p + dyn_p

        if self.temperature > self.dp.t_throttle:
            power *= 0.85
        if self.temperature > self.dp.t_hardcap:
            power *= 0.7

        return power

    def thermal_update(self, dt: float):
        """
        Improved first-order RC-like thermal dynamics.
        - Includes stochastic variation (cooling noise)
        - Models thermal inertia (slow response)
        - Allows visible heating/cooling cycles
        """
        heat_gain = self.dp.t_rise_gain * self.power
        cooling = self.dp.t_cool_gain * (self.temperature - self.dp.t_ambient)
        noise = 0.5 * self.rng.standard_normal()

        dT = (heat_gain - cooling) + 0.05 * noise
        self.temperature += dT * (dt * 15.0)

        self.temperature = clamp(self.temperature, self.dp.t_ambient - 5.0, 100.0)

        if self.temperature > self.dp.t_throttle:
            self.latency_ms += 0.05 * (self.temperature - self.dp.t_throttle)

    def map_states_to_latency(self, util):
        """
        U-shaped latency model:
        - low util: GPU underclock → higher latency
        - mid util: optimal region → lowest latency
        - high util: overload → latency increases again
        """

        # GPU optimal utilization point
        u_opt = 0.55  # sweet spot for VR pipeline

        # U-shaped latency curve
        u_curve = (util - u_opt) ** 2

        # Base latency
        lat = self.dp.latency_base_ms * self.ht["l_base_mul"]

        # Util-dependent latency (U shape)
        lat += self.dp.k_latency_util * u_curve * self.ht["l_util_mul"]

        # Task-load latency
        lat += self.dp.k_latency_tasks * self.task_load * self.ht["l_task_mul"]

        # Power / thermal effect
        #  (instead of monotonic increase, we give a soft penalty)
        thermal_penalty = max(0.0, (self.temperature - self.dp.t_throttle))
        lat += 0.02 * thermal_penalty

        # Small jitter from power fluctuation
        lat += 0.002 * max(0, self.power - self.dp.power_base)

        return lat

    def map_to_quality(self, latency, motion, jitter, q):
        lat_penalty = 1/(1+0.02*max(0, latency))
        jit_penalty = 1/(1+self.dp.jitter_sensitivity*max(0, jitter))
        mot_penalty = 1/(1+self.dp.motion_sensitivity*max(0, motion))
        return clamp(q * lat_penalty * jit_penalty * mot_penalty, 0, 1)

    def control_loop(self):
        dt = self.gp.control_dt
        while True:
            t = self.env.now
            self.update_exogenous(t)
            util_cap = self.util_allocation

            if self.offline:
                self.util = 0.0
                self.power = 0.0
                self.latency_ms = 0.0
                self.quality = 0.0
                self.temperature -= 0.2 * dt * 50
                self.temperature = max(self.dp.t_ambient - 5, self.temperature)
                t = self.env.now
                self.logs["t"].append(t)
                for k in ["q", "util", "latency", "power", "quality", "motion", "taskload", "temp"]:
                    if k == "temp":
                        self.logs[k].append(self.temperature)
                    else:
                        self.logs[k].append(0)
                yield self.env.timeout(dt)
                continue

            util_demand = self.map_q_to_util(self.q)

            util_demand *= self.external_load_factor
            util_demand = clamp(util_demand, 0.0, 1.0)

            self.util = min(util_demand, util_cap)

            self.power = self.map_util_to_power(self.util)
            self.thermal_update(dt)
            latency = self.map_states_to_latency(self.util)
            self.latency_ms_avg = lowpass(self.latency_ms_avg, latency, 0.3)
            jitter = abs(latency - self.latency_ms_avg)
            self.quality = self.map_to_quality(latency, self.motion, jitter, self.q)

            err = latency - self.dp.target_latency_ms
            self.q_int += err * dt
            dq = -(self.dp.pid_kp*err + self.dp.pid_ki*self.q_int)
            dq = clamp(dq, -self.dp.q_rate_limit*dt, self.dp.q_rate_limit*dt)
            self.q = clamp(self.q + dq, 0.1, 1.0)

            self.latency_ms = latency
            self.logs["t"].append(t)
            for k,v in [("q",self.q),("util",self.util),("latency",latency),
                        ("power",self.power),("quality",self.quality),
                        ("motion",self.motion),("taskload",self.task_load),
                        ("temp",self.temperature)]:
                self.logs[k].append(v)
            yield self.env.timeout(dt)
